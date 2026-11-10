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

// The epoch's sequence order: sequences of similar length are grouped -- sorted by length and cut
// into buckets of 32 -- and each bucket is then shuffled with the epoch's rng. Grouping first is
// what keeps a step from mixing one long sequence with many short ones; shuffling inside the
// bucket is what makes the order a function of (seed, epoch), including when the whole dataset
// fits in a single bucket, where a shuffle followed by a sort by length would leave no shuffle at
// all. (A sort and then a shuffle, rather than the other way round, is therefore deliberate.)
std::vector<size_t> epoch_order(const std::vector<size_t>& kept, uint64_t seed, int64_t epoch,
                                const std::vector<int64_t>& lens) {
    std::vector<size_t> order = kept;
    // Descending by length, input index to break ties: a total order, so the grouping is the same
    // on every platform.
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (lens[a] != lens[b]) return lens[a] > lens[b];
        return a < b;
    });
    constexpr size_t kBucket = 32;
    uint64_t rng = epoch_rng(seed, epoch);
    for (size_t lo = 0; lo < order.size(); lo += kBucket) {
        size_t hi = std::min(order.size(), lo + kBucket);
        for (size_t i = hi; i > lo + 1; --i) {
            size_t j = lo + (size_t)(splitmix64(&rng) % (i - lo));
            std::swap(order[i - 1], order[j]);
        }
    }
    return order;
}

// A step's sequences -> one sub-batch per device with balanced token counts. Longest first, each
// sequence handed to the least-loaded device; this keeps the spread below the longest sequence of
// the step. (A spread of exactly one maximal sequence can be forced -- one sequence for two
// devices, or three budget-length sequences into two devices -- and no arrangement does better.)
std::vector<std::vector<size_t>> balance(const std::vector<size_t>& idx,
                                         const std::vector<int64_t>& lens, int devices) {
    std::vector<size_t> sorted = idx;
    std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) {
        if (lens[a] != lens[b]) return lens[a] > lens[b];
        return a < b;
    });
    std::vector<std::vector<size_t>> subs((size_t)devices);
    std::vector<int64_t> loads((size_t)devices, 0);
    for (size_t s : sorted) {
        size_t d = (size_t)(std::min_element(loads.begin(), loads.end()) - loads.begin());
        subs[d].push_back(s);
        loads[d] += lens[s];
    }
    // Keep a stable order inside each sub-batch: longest first, input index to break ties.
    for (auto& sub : subs)
        std::sort(sub.begin(), sub.end(), [&](size_t a, size_t b) {
            if (lens[a] != lens[b]) return lens[a] > lens[b];
            return a < b;
        });
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
