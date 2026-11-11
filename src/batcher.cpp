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

// The bucket an epoch starts on. The seed chooses where in the bucket cycle it starts and every
// epoch moves that on by one, so two consecutive epochs start on different buckets; a bucket is
// never empty, so that alone is a different order of the sequences, whatever the shuffle draws.
// Rotating a fresh shuffle by an offset cannot promise the same: the shuffle and the offset are
// drawn independently, so an epoch's order lands on the previous epoch's whenever the two shuffles
// are rotations of each other. Nor may the offset be a linear function of the seed, as `3 * seed`
// was -- it vanishes whenever the bucket count divides three times the seed, and two seeds then
// hand out one order. The seed enters here as a hash.
size_t starting_bucket(uint64_t seed, int64_t epoch, size_t buckets) {
    uint64_t state = seed ^ 0xa0761d6478bd642full;
    const size_t base = (size_t)(splitmix64(&state) % (uint64_t)buckets);
    return (base + (size_t)((uint64_t)epoch % (uint64_t)buckets)) % buckets;
}

}  // namespace

namespace {

// The epoch's buckets: sequences of similar length are grouped -- sorted by token count and cut
// into buckets of at most 32, where the power-of-two length class changes -- and then the buckets
// themselves are shuffled with the epoch's rng. The class cut is what keeps a step from mixing one
// long sequence with many short ones: inside a bucket no two lengths differ by a factor of two or
// more. Shuffling the buckets is what makes the order a function of (seed, epoch) even when every
// bucket holds a single sequence, which a shuffle inside the buckets could never reach.
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
        buckets.emplace_back(order.begin() + static_cast<std::ptrdiff_t>(lo),
                             order.begin() + static_cast<std::ptrdiff_t>(hi));
        lo = hi;
    }
    // Inside a bucket the order is a shuffle too, drawn from a stream for (seed, epoch, bucket).
    // The sort did its work when it formed the buckets -- inside one, every length is in one
    // power-of-two class -- so shuffling them costs the grouping nothing, and it is what gives the
    // seed and the epoch room to differ: the bucket order alone is n! orders for n buckets, so two
    // seeds on a dataset of two or three classes would hand out the same batches one time in two
    // or six, however the seed were hashed into the choice.
    if (buckets.size() > 1) {
        for (size_t b = 0; b < buckets.size(); ++b) {
            uint64_t rng =
                epoch_rng(seed ^ (0x2545f4914f6cdd1dull * (uint64_t)(b + 1)), epoch);
            for (size_t i = buckets[b].size(); i > 1; --i) {
                size_t j = (size_t)(splitmix64(&rng) % (uint64_t)i);
                std::swap(buckets[b][i - 1], buckets[b][j]);
            }
        }
        // Shuffle the buckets' positions, then start the order on the bucket the epoch names. The
        // start is what makes each epoch a new shuffle rather than a fresh draw that may land on
        // the previous one: consecutive epochs start on different buckets, so no two of them can
        // hand out the same order. The rest of the order is the shuffled one, so the seed and the
        // epoch still reach every bucket, and a dataset of singletons still moves.
        const size_t n = buckets.size();
        std::vector<size_t> positions(n);
        for (size_t i = 0; i < n; ++i) positions[i] = i;
        uint64_t rng = epoch_rng(seed, epoch);
        for (size_t i = n; i > 1; --i) {
            size_t j = (size_t)(splitmix64(&rng) % (uint64_t)i);
            std::swap(positions[i - 1], positions[j]);
        }
        const size_t first = starting_bucket(seed, epoch, n);
        std::rotate(positions.begin(), std::find(positions.begin(), positions.end(), first),
                    positions.end());
        std::vector<std::vector<size_t>> shuffled;
        shuffled.reserve(n);
        for (size_t i : positions) shuffled.push_back(std::move(buckets[i]));
        buckets = std::move(shuffled);
    } else if (!buckets.empty()) {
        // One bucket is no bucket order to shuffle, so every epoch's order has to come from inside
        // that one bucket. It is shuffled under the seed alone and then rotated by the epoch: two
        // rotations of one permutation by different shifts are different orders, so consecutive
        // epochs differ even when there are only two sequences. (Shuffling it again per epoch is
        // no guarantee: two such shuffles can come out one rotation apart.)
        auto& bucket = buckets.front();
        uint64_t single = epoch_rng(seed, 0);
        for (size_t i = bucket.size(); i > 1; --i) {
            size_t j = (size_t)(splitmix64(&single) % i);
            std::swap(bucket[i - 1], bucket[j]);
        }
        if (bucket.size() > 1) {
            const size_t k = bucket.size();
            const size_t shift = (size_t)((uint64_t)epoch % k);
            std::rotate(bucket.begin(), bucket.begin() + static_cast<std::ptrdiff_t>(shift),
                        bucket.end());
        }
    }
    return buckets;
}

using Subs = std::vector<std::vector<size_t>>;
using PlannedSteps = std::vector<Subs>;

// The packer walks the buckets in their shuffled order and keeps one step open across a bucket
// edge. Walking that order, each sequence goes to the least-loaded sub-batch that can still hold
// it under the budget (ties to the lowest device); when none can, the step closes and the next
// sequence starts a new one. So every sub-batch stays under the budget and a step may spend up to
// devices * budget.
//
// A sub-batch is only ever added to when the step it is in still hands out sub-batches within one
// maximal sequence of each other in tokens: the spread is part of what a device is asked for, so
// an add that would break it is no candidate at all. A step then closes with its devices full --
// every device is either carrying a sub-batch or was sealed by an earlier class -- and a sub-batch
// that no longer has room is what closes one early. The sequences a step could not take are the
// first of the next step, not a step of their own: nothing is split off after the fact, which
// would hand out a step of one sub-batch in the middle of an epoch.
//
// A bucket that ends with fewer sequences than devices leaves idle devices. Those sequences are
// carried into the next bucket instead of being handed out with an empty sub-batch. If the next
// bucket is another chunk of the same length class, it simply continues filling the step; at a
// class edge the occupied devices are sealed (the next class must not join them and mix classes
// in one sub-batch), and the empty devices are filled from the next bucket. So a step can hold two
// classes in different sub-batches, which is how a short tail fills a device a long sequence left
// idle. Only the epoch's last tail has nothing to carry into, and there the step hands out the
// sub-batches it has, never an empty one.
struct StepPacker {
    const std::vector<int64_t>& lens;
    int64_t budget;
    int devices;
    Subs subs;
    std::vector<int64_t> loads;
    std::vector<char> sealed;  // occupied by an earlier bucket; the current bucket may not add
    int64_t step_max = 0;      // the longest sequence the open step holds
    PlannedSteps steps;
    static constexpr size_t kNone = (size_t)-1;

    StepPacker(const std::vector<int64_t>& lens_, int64_t budget_, int devices_)
        : lens(lens_), budget(budget_), devices(devices_),
          subs((size_t)devices_), loads((size_t)devices_, 0), sealed((size_t)devices_, 0) {}

    void reset() {
        for (auto& sub : subs) sub.clear();
        std::fill(loads.begin(), loads.end(), int64_t{0});
        std::fill(sealed.begin(), sealed.end(), char{0});
        step_max = 0;
    }

    void close() {
        Subs step;
        for (auto& sub : subs)
            if (!sub.empty()) step.push_back(std::move(sub));
        if (!step.empty()) steps.push_back(std::move(step));
        reset();
    }

    // Whether adding `s` to device `d` keeps the step's sub-batches within one maximal sequence of
    // each other in tokens. The spread is over the sub-batches the step hands out, so a step that
    // would hand out only one of them has nothing to bound, and a step whose sequences are all of
    // no tokens has no sequence to bound it by.
    bool keeps_spread(size_t s, size_t d) const {
        const int64_t grown = loads[d] + lens[s];
        int64_t hi = grown, lo = grown;
        size_t nonempty = 1;
        for (size_t k = 0; k < (size_t)devices; ++k) {
            if (k == d || subs[k].empty()) continue;
            hi = std::max(hi, loads[k]);
            lo = std::min(lo, loads[k]);
            ++nonempty;
        }
        if (nonempty < 2) return true;
        const int64_t max_seq = std::max(step_max, lens[s]);
        return max_seq == 0 || hi - lo < max_seq;
    }

    size_t pick(size_t s) const {
        size_t best = kNone;
        for (size_t k = 0; k < (size_t)devices; ++k) {
            if (sealed[k]) continue;
            if (loads[k] + lens[s] > budget) continue;
            if (!keeps_spread(s, k)) continue;
            if (best == kNone || loads[k] < loads[best]) best = k;
        }
        return best;
    }

    void add(size_t s, size_t d) {
        subs[d].push_back(s);
        loads[d] += lens[s];
        step_max = std::max(step_max, lens[s]);
    }

    void pack_bucket(const std::vector<size_t>& bucket, bool has_next, bool next_same_class) {
        for (size_t s : bucket) {
            size_t d = pick(s);
            if (d == kNone) {
                close();
                d = pick(s);  // every sub-batch is empty again, and a kept sequence fits one
            }
            add(s, d);
        }
        size_t nonempty = 0;
        for (const auto& sub : subs)
            if (!sub.empty()) ++nonempty;
        if (nonempty == 0) return;
        if (has_next && (int)nonempty < devices && !next_same_class) {
            // Carry this short step into the next class: keep what it has, open what it lacks.
            for (size_t k = 0; k < (size_t)devices; ++k) sealed[k] = !subs[k].empty();
        } else if (!has_next || !next_same_class || (int)nonempty == devices) {
            // A same-class bucket edge keeps the step open: the next chunk may fill it without
            // mixing classes, so it is not a tail to carry across a sealed boundary.
            close();
        }
    }
};

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
    // A step normally never crosses a bucket edge, so classes do not meet in one sub-batch. The
    // exception is a bucket that ends with fewer sequences than devices: those sequences carry
    // forward to fill the next bucket's step instead of leaving empty devices behind.
    StepPacker packer(lens, config_.token_budget, config_.devices);
    for (size_t i = 0; i < buckets.size(); ++i) {
        const bool has_next = i + 1 < buckets.size();
        const bool next_same_class =
            has_next && std::bit_width((uint64_t)lens[buckets[i][0]]) ==
                             std::bit_width((uint64_t)lens[buckets[i + 1][0]]);
        packer.pack_bucket(buckets[i], has_next, next_same_class);
    }
    EpochPlan p;
    p.steps = std::move(packer.steps);
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
    if (position_.step < 0 || position_.step >= (int64_t)p.steps.size()) {
        // seek() clamps every position into the plan, but next() must not trust one either: leave
        // it at the epoch's end instead of indexing outside the plan.
        position_.step = (int64_t)p.steps.size();
        return false;
    }
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

void Batcher::seek(const Position& p) {
    // A position comes back from stored data, not from an index expression: clamp it into the
    // valid range so a stepped-back, corrupted or out-of-range one cannot index the plan out of
    // bounds. The epoch's end is a valid position (next() there returns false), hence inclusive.
    position_.epoch = std::max<int64_t>(p.epoch, 0);
    position_.step = std::clamp<int64_t>(p.step, 0, steps_in_epoch(position_.epoch));
}

const std::vector<size_t>& Batcher::skipped() const { return skipped_; }

int64_t Batcher::skipped_tokens() const { return skipped_tokens_; }

const std::vector<TokenSeq>& Batcher::sequences() const { return sequences_; }

const BatcherConfig& Batcher::config() const { return config_; }

}  // namespace r97
