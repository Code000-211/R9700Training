// r97/batcher.cpp -- the token-budget batcher. See r97/batcher.h and docs/batcher.md.
#include "r97/batcher.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace r97 {

namespace {

int64_t token_count(const TokenSeq& s) { return (int64_t)s.ids.size(); }

// splitmix64: the whole determinism rests on this and a Fisher-Yates over it.
uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// The epoch's rng: one warm-up so nearby (seed, epoch) pairs decorrelate.
uint64_t epoch_rng(uint64_t seed, int64_t epoch) {
    uint64_t state = seed ^ (0xd1b54a32d192ed03ull + (uint64_t)epoch * 0x9e3779b97f4a7c15ull);
    splitmix64(&state);
    return state;
}

}  // namespace

namespace {

// The epoch's sequence order: a fresh shuffle, then sequences of similar length grouped -- the
// shuffled list is cut into buckets of 32 sequences and each bucket is sorted by length, so a
// batch never mixes one long sequence with many short ones. Deterministic for (seed, epoch).
std::vector<size_t> epoch_order(const std::vector<size_t>& kept, uint64_t seed, int64_t epoch,
                                const std::vector<int64_t>& lens) {
    std::vector<size_t> order = kept;
    uint64_t rng = epoch_rng(seed, epoch);
    for (size_t i = order.size(); i > 1; --i) {
        size_t j = (size_t)(splitmix64(&rng) % i);
        std::swap(order[i - 1], order[j]);
    }
    constexpr size_t kBucket = 32;
    for (size_t lo = 0; lo < order.size(); lo += kBucket) {
        size_t hi = std::min(order.size(), lo + kBucket);
        std::sort(order.begin() + (long)lo, order.begin() + (long)hi,
                  [&](size_t a, size_t b) { return lens[a] > lens[b]; });
    }
    return order;
}

// A step's sequences -> one sub-batch per device with balanced token counts. Sorted longest first
// and dealt out in snake order (device 0..N-1, then back), which spreads similar lengths evenly;
// a repair pass then moves the smallest sequence from the fullest to the emptiest device while the
// spread is not below one maximal sequence. (A spread of exactly one maximal sequence can be
// forced -- three budget-length sequences into two devices -- and no arrangement does better.)
std::vector<std::vector<size_t>> balance(const std::vector<size_t>& idx,
                                         const std::vector<int64_t>& lens, int devices) {
    std::vector<size_t> sorted = idx;
    std::sort(sorted.begin(), sorted.end(),
              [&](size_t a, size_t b) { return lens[a] > lens[b]; });
    std::vector<std::vector<size_t>> subs((size_t)devices);
    std::vector<int64_t> loads((size_t)devices, 0);
    int d = 0, dir = 1;
    for (size_t s : sorted) {
        subs[(size_t)d].push_back(s);
        loads[(size_t)d] += lens[s];
        int nd = d + dir;
        if (nd < 0 || nd == devices) { dir = -dir; nd = d + dir; }
        d = nd;
    }
    const int64_t max_seq = lens[sorted[0]];
    for (;;) {
        size_t mx = (size_t)(std::max_element(loads.begin(), loads.end()) - loads.begin());
        size_t mn = (size_t)(std::min_element(loads.begin(), loads.end()) - loads.begin());
        if (loads[mx] - loads[mn] < max_seq) break;
        // The smallest sequence of the fullest device, moved over: strict improvement or stop.
        auto it = std::min_element(subs[mx].begin(), subs[mx].end(),
                                   [&](size_t a, size_t b) { return lens[a] < lens[b]; });
        int64_t len = lens[*it];
        int64_t spread = loads[mx] - loads[mn];
        // spread' = max(load anywhere) - (loads[mn] + len)
        int64_t new_max = 0;
        for (size_t k = 0; k < subs.size(); ++k) {
            if (k == mx) continue;
            new_max = std::max(new_max, loads[k]);
        }
        new_max = std::max({new_max, loads[mx] - len, loads[mn] + len});
        int64_t new_min = loads[mn] + len;
        for (size_t k = 0; k < subs.size(); ++k) {
            if (k == mn) continue;
            new_min = std::min(new_min, k == mx ? loads[mx] - len : loads[k]);
        }
        if (new_max - new_min >= spread) break;
        subs[mn].push_back(*it);
        subs[mx].erase(it);
        loads[mn] += len;
        loads[mx] -= len;
    }
    // Keep a stable order inside each sub-batch: longest first.
    for (auto& sub : subs)
        std::sort(sub.begin(), sub.end(), [&](size_t a, size_t b) { return lens[a] > lens[b]; });
    return subs;
}

}  // namespace

const Batcher::EpochPlan& Batcher::plan(int64_t epoch) const {
    if (planned_epoch_ == epoch) return cached_plan_;
    std::vector<int64_t> lens(sequences_.size());
    for (size_t i = 0; i < sequences_.size(); ++i) lens[i] = token_count(sequences_[i]);
    std::vector<size_t> kept;
    kept.reserve(sequences_.size() - skipped_.size());
    const size_t* skip = skipped_.data();
    size_t nskip = skipped_.size();
    for (size_t i = 0; i < sequences_.size(); ++i) {
        bool over = nskip > 0 && skip[0] == i;
        if (over) { ++skip; --nskip; continue; }
        kept.push_back(i);
    }
    std::vector<size_t> order = epoch_order(kept, config_.seed, epoch, lens);
    EpochPlan p;
    std::vector<size_t> cur;
    int64_t total = 0;
    auto close_step = [&]() {
        if (cur.empty()) return;
        p.steps.push_back(balance(cur, lens, config_.devices));
        cur.clear();
        total = 0;
    };
    for (size_t s : order) {
        if (!cur.empty() && total + lens[s] > config_.token_budget) close_step();
        cur.push_back(s);
        total += lens[s];
    }
    close_step();
    cached_plan_ = std::move(p);
    planned_epoch_ = epoch;
    return cached_plan_;
}

Batcher::Batcher(std::vector<TokenSeq> sequences, const BatcherConfig& config)
    : sequences_(std::move(sequences)), config_(config) {
    if (config_.token_budget <= 0)
        throw std::invalid_argument("batcher: token_budget must be positive");
    if (config_.devices < 1) throw std::invalid_argument("batcher: devices must be at least 1");
    for (size_t i = 0; i < sequences_.size(); ++i) {
        int64_t n = token_count(sequences_[i]);
        if (n > config_.token_budget) {
            skipped_.push_back(i);
            skipped_tokens_ += n;
        }
    }
}

bool Batcher::next(std::vector<Batch>* step) {
    const EpochPlan& p = plan(position_.epoch);
    if (position_.step >= (int64_t)p.steps.size()) return false;
    const auto& idx = p.steps[(size_t)position_.step];
    step->clear();
    for (size_t d = 0; d < idx.size(); ++d) {
        Batch b;
        b.reserve(idx[d].size());
        for (size_t s : idx[d]) b.push_back(sequences_[s]);
        step->push_back(std::move(b));
    }
    ++position_.step;
    return true;
}

int64_t Batcher::steps_in_epoch(int64_t epoch) const { return (int64_t)plan(epoch).steps.size(); }

const Batcher::Position& Batcher::position() const { return position_; }

void Batcher::reset() { position_ = Position{}; }

void Batcher::seek(const Position& p) { position_ = p; }

const std::vector<size_t>& Batcher::skipped() const { return skipped_; }

int64_t Batcher::skipped_tokens() const { return skipped_tokens_; }

const std::vector<TokenSeq>& Batcher::sequences() const { return sequences_; }

const BatcherConfig& Batcher::config() const { return config_; }

}  // namespace r97
