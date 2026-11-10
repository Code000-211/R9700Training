// r97/batcher.cpp -- the token-budget batcher. See r97/batcher.h and docs/batcher.md.
#include "r97/batcher.h"

#include <algorithm>
#include <bit>
#include <cstddef>
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

// The epoch's buckets: sequences of similar length are grouped -- sorted by token count and cut
// into buckets of at most 32, where the power-of-two length class changes -- and each bucket is
// then shuffled with the epoch's rng. The class cut is what keeps a step from mixing one long
// sequence with many short ones: inside a bucket no two lengths differ by a factor of two or more.
// The 32 cap keeps the buckets small enough that the shuffle inside one has a fine effect on the
// steps' composition. Shuffling is what makes the order a function of (seed, epoch), including
// when the whole dataset fits in a single bucket, where a shuffle followed by a sort by length
// would leave no shuffle at all. (A sort and then a shuffle, rather than the other way round, is
// therefore deliberate.)
std::vector<std::vector<size_t>> epoch_buckets(const std::vector<size_t>& kept, uint64_t seed,
                                               int64_t epoch,
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
    std::vector<std::vector<size_t>> buckets;
    size_t lo = 0;
    while (lo < order.size()) {
        // std::bit_width is the length's power-of-two class: equal for lengths in [2^k, 2^(k+1)),
        // so every pair inside a bucket is within a factor of two. Zero-length sequences share
        // class 0. The sort keeps the class nonincreasing, so a change is always a cut.
        const int cls = (int)std::bit_width((uint64_t)lens[order[lo]]);
        size_t hi = lo;
        while (hi < order.size() && hi - lo < kBucket &&
               (int)std::bit_width((uint64_t)lens[order[hi]]) == cls)
            ++hi;
        for (size_t i = hi; i > lo + 1; --i) {
            size_t j = lo + (size_t)(splitmix64(&rng) % (i - lo));
            std::swap(order[i - 1], order[j]);
        }
        buckets.emplace_back(order.begin() + static_cast<std::ptrdiff_t>(lo),
                             order.begin() + static_cast<std::ptrdiff_t>(hi));
        lo = hi;
    }
    return buckets;
}

// A bucket's shuffled sequences -> steps of one sub-batch per device. Walking the shuffled order,
// each sequence goes to the least-loaded sub-batch that can still hold it under the budget (ties
// to the lowest device); when none can, the step closes and the next sequence starts a new one. So
// every sub-batch stays under the budget, a step may spend up to devices * budget, and the spread
// between a step's fullest and emptiest sub-batch never exceeds the step's longest sequence. A
// step with fewer sequences than devices -- a bucket's end can leave one -- has an empty
// sub-batch and a spread of exactly one maximal sequence; no packing can avoid that.
std::vector<std::vector<std::vector<size_t>>> pack_bucket(
    const std::vector<size_t>& bucket, const std::vector<int64_t>& lens, int64_t budget,
    int devices) {
    std::vector<std::vector<std::vector<size_t>>> steps;
    std::vector<std::vector<size_t>> subs((size_t)devices);
    std::vector<int64_t> loads((size_t)devices, 0);
    const size_t kNone = (size_t)devices;
    auto close_step = [&]() {
        bool empty = true;
        for (const auto& sub : subs)
            if (!sub.empty()) {
                empty = false;
                break;
            }
        if (empty) return;
        // Keep a stable order inside each sub-batch: longest first, input index to break ties.
        for (auto& sub : subs)
            std::sort(sub.begin(), sub.end(), [&](size_t a, size_t b) {
                if (lens[a] != lens[b]) return lens[a] > lens[b];
                return a < b;
            });
        steps.push_back(subs);
        for (auto& sub : subs) sub.clear();
        std::fill(loads.begin(), loads.end(), int64_t{0});
    };
    for (size_t s : bucket) {
        size_t d = kNone;
        for (size_t k = 0; k < (size_t)devices; ++k) {
            if (loads[k] + lens[s] > budget) continue;
            if (d == kNone || loads[k] < loads[d]) d = k;
        }
        if (d == kNone) {
            close_step();
            d = 0;  // every sub-batch is empty again, and a kept sequence fits one
        }
        subs[d].push_back(s);
        loads[d] += lens[s];
    }
    close_step();
    return steps;
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
    std::vector<std::vector<size_t>> buckets = epoch_buckets(kept, config_.seed, epoch, lens);
    EpochPlan p;
    for (const std::vector<size_t>& bucket : buckets) {
        // A step never crosses a bucket edge, so the classes never meet in one step.
        auto steps = pack_bucket(bucket, lens, config_.token_budget, config_.devices);
        for (auto& step : steps) p.steps.push_back(std::move(step));
    }
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
