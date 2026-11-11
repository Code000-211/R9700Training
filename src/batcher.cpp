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
    // Shuffle the bucket order, not the sequences inside: the sort within a bucket is what keeps
    // similar lengths together, and the bucket is the unit the epoch varies. This is also what a
    // bucket of one needs -- there is nothing inside it to reorder.
    uint64_t rng = epoch_rng(seed, epoch);
    for (size_t i = buckets.size(); i > 1; --i) {
        size_t j = (size_t)(splitmix64(&rng) % i);
        std::swap(buckets[i - 1], buckets[j]);
    }
    // Fisher-Yates can hand out the same order for nearby seeds when there are only a couple of
    // buckets, and can repeat a consecutive epoch's order by chance. Rotate the shuffled order by
    // an offset that the seed shifts and each epoch advances: a bucket of one still moves with
    // (seed, epoch), and the common short-bucket collisions are broken. 3 is odd, so for two
    // buckets the seed still flips the order.
    if (buckets.size() > 1) {
        const size_t n = buckets.size();
        const size_t shift = (size_t)(((uint64_t)epoch + 3 * seed) % n);
        std::rotate(buckets.begin(), buckets.begin() + static_cast<std::ptrdiff_t>(shift),
                    buckets.end());
    } else if (!buckets.empty()) {
        // One bucket is no bucket order to shuffle. Sorting it by length would leave the seed and
        // the epoch with nothing to change on a dataset that fits one bucket, so shuffle its
        // sequences under the seed and then rotate by the epoch. The rotation makes every epoch's
        // order a new one even when there are only two sequences.
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
// A bucket that ends with fewer sequences than devices leaves idle devices. Those sequences are
// carried into the next bucket instead of being handed out with an empty sub-batch. If the next
// bucket is another chunk of the same length class, it simply continues filling the step; at a
// class edge the occupied devices are sealed (the next class must not join them and mix classes
// in one sub-batch), and the empty devices are filled from the next bucket. Only the epoch's last
// tail has nothing to carry into, and there the step hands out the sub-batches it has, never an
// empty one.
struct StepPacker {
    const std::vector<int64_t>& lens;
    int64_t budget;
    int devices;
    Subs subs;
    std::vector<int64_t> loads;
    std::vector<char> sealed;  // occupied by an earlier bucket; the current bucket may not add
    PlannedSteps steps;
    static constexpr size_t kNone = (size_t)-1;

    StepPacker(const std::vector<int64_t>& lens_, int64_t budget_, int devices_)
        : lens(lens_), budget(budget_), devices(devices_),
          subs((size_t)devices_), loads((size_t)devices_, 0), sealed((size_t)devices_, 0) {}

    void reset() {
        for (auto& sub : subs) sub.clear();
        std::fill(loads.begin(), loads.end(), int64_t{0});
        std::fill(sealed.begin(), sealed.end(), char{0});
    }

    void close() {
        bool empty = true;
        for (const auto& sub : subs)
            if (!sub.empty()) {
                empty = false;
                break;
            }
        if (!empty) steps.push_back(subs);
        reset();
    }

    size_t pick(size_t s) const {
        size_t best = kNone;
        for (size_t k = 0; k < (size_t)devices; ++k) {
            if (sealed[k]) continue;
            if (loads[k] + lens[s] > budget) continue;
            if (best == kNone || loads[k] < loads[best]) best = k;
        }
        return best;
    }

    void add(size_t s, size_t d) {
        subs[d].push_back(s);
        loads[d] += lens[s];
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

// One closed step -> the steps it becomes. Empty sub-batches are dropped, and a step whose
// sub-batches differ by at least its longest sequence is split: the fullest sub-batch gives up
// its last sequences to a step of its own until the remainder is balanced. What comes off is one
// sub-batch's worth, which is under the budget, so the spill is a valid sub-batch. The result is
// the strict bound the packing wants -- less than one maximal sequence between sub-batches -- and
// a step of fewer than two sub-batches (an epoch's final tail) has no spread to bound.
PlannedSteps repair_step(Subs subs, const std::vector<int64_t>& lens) {
    subs.erase(std::remove_if(subs.begin(), subs.end(),
                              [](const std::vector<size_t>& sub) { return sub.empty(); }),
               subs.end());
    PlannedSteps spills;
    while (subs.size() >= 2) {
        std::vector<int64_t> loads(subs.size(), 0);
        int64_t max_seq = 0;
        for (size_t d = 0; d < subs.size(); ++d) {
            for (size_t s : subs[d]) {
                loads[d] += lens[s];
                max_seq = std::max(max_seq, lens[s]);
            }
        }
        int64_t lo = loads[0], hi = loads[0];
        size_t full = 0;
        for (size_t d = 1; d < subs.size(); ++d) {
            lo = std::min(lo, loads[d]);
            if (loads[d] > hi) {
                hi = loads[d];
                full = d;
            }
        }
        if (max_seq == 0 || hi - lo < max_seq) break;

        std::vector<size_t> spill;
        while (!subs[full].empty()) {
            spill.push_back(subs[full].back());
            subs[full].pop_back();
            // Recompute over the sub-batches that still hold something.
            std::vector<int64_t> rest_loads;
            int64_t rest_max = 0;
            for (const auto& sub : subs) {
                if (sub.empty()) continue;
                int64_t load = 0;
                for (size_t s : sub) {
                    load += lens[s];
                    rest_max = std::max(rest_max, lens[s]);
                }
                rest_loads.push_back(load);
            }
            if (rest_loads.size() < 2) break;
            int64_t rest_lo = rest_loads[0], rest_hi = rest_loads[0];
            for (int64_t load : rest_loads) {
                rest_lo = std::min(rest_lo, load);
                rest_hi = std::max(rest_hi, load);
            }
            if (rest_max == 0 || rest_hi - rest_lo < rest_max) break;
        }
        subs.erase(std::remove_if(subs.begin(), subs.end(),
                                  [](const std::vector<size_t>& sub) { return sub.empty(); }),
                   subs.end());
        if (!spill.empty()) {
            std::reverse(spill.begin(), spill.end());  // back into the shuffle's order
            spills.push_back(Subs{std::move(spill)});
        }
    }

    PlannedSteps out;
    if (!subs.empty()) out.push_back(std::move(subs));
    for (auto& spill : spills) out.push_back(std::move(spill));
    return out;
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
    for (auto& step : packer.steps) {
        PlannedSteps fixed = repair_step(std::move(step), lens);
        for (auto& one : fixed) p.steps.push_back(std::move(one));
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
