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

// The packer walks the merged runs of the shuffled buckets and keeps one step open across a run.
// Walking that order, each sequence goes to the least-loaded sub-batch that can still hold it
// under the budget (ties to the lowest device); when none can, the step closes and the next
// sequence starts a new one. So every sub-batch stays under the budget and a step may spend up to
// devices * budget.
//
// The spread is a property of the step, not of every intermediate state of the packing. A step
// that fills four sub-batches of equal length passes through states whose spread is exactly one
// sequence, on its way from one sequence per sub-batch to four; bounding every add would stop it
// after the first sequence and spend a fraction of the budget. So an add that lands the spread on
// the bound is admitted while the rest of the run can still raise the sub-batches at the bottom to
// meet the one at the top. When it cannot -- the run ends first, or the next sequences do not fit
// -- the add is refused and the step closes on the state before it, which is still inside the
// bound. Thus every closed step carries its sub-batches within one maximal sequence of each other,
// and the check runs forwards over the next sequences rather than backwards over a packed step.
//
// A run that ends with fewer sequences than devices leaves idle devices. Those sequences are
// carried into the next run instead of being handed out with an empty sub-batch. At a class edge
// the occupied devices are sealed (the next class must not join them and mix classes in one
// sub-batch), and the empty devices are filled from the next run. So a step can hold two classes
// in different sub-batches, which is how a short tail fills a device a long sequence left idle.
// Only the epoch's last tail has nothing to carry into, and there the step hands out the
// sub-batches it has, never an empty one.
struct StepPacker {
    const std::vector<int64_t>& lens;
    int64_t budget;
    int devices;
    Subs subs;
    std::vector<int64_t> loads;
    std::vector<char> sealed;  // occupied by an earlier bucket; the current bucket may not add
    std::vector<int> classes;  // the length class of each sub-batch; -1 while it is empty
    int64_t step_max = 0;      // the longest sequence the open step holds
    PlannedSteps steps;
    static constexpr size_t kNone = (size_t)-1;

    StepPacker(const std::vector<int64_t>& lens_, int64_t budget_, int devices_)
        : lens(lens_), budget(budget_), devices(devices_), subs((size_t)devices_),
          loads((size_t)devices_, 0), sealed((size_t)devices_, 0),
          classes((size_t)devices_, -1) {}

    void reset() {
        for (auto& sub : subs) sub.clear();
        std::fill(loads.begin(), loads.end(), int64_t{0});
        std::fill(sealed.begin(), sealed.end(), char{0});
        std::fill(classes.begin(), classes.end(), -1);
        step_max = 0;
    }

    void close() {
        Subs step;
        for (auto& sub : subs)
            if (!sub.empty()) step.push_back(std::move(sub));
        if (!step.empty()) steps.push_back(std::move(step));
        reset();
    }

    // Whether the step after adding `s` to `d` can reach a spread inside `max_seq` using the
    // sequences that follow in the run. The step is at the bound after the add, so every sub-batch
    // at the bottom has to be raised before the step closes; raising one costs one of the run's
    // next sequences, and a device at the bottom accepts a sequence of the step's maximal length
    // while the step is under the budget. At most `devices` adds are needed. Only a step whose
    // sub-batches are all of one class is simulated: a carried class edge seals its sub-batches,
    // so the bottom may be a device that no later sequence can raise.
    bool rebalances(size_t s, size_t d, const std::vector<size_t>& run, size_t next) const {
        std::vector<int64_t> future_loads = loads;
        std::vector<char> used((size_t)devices, 0);
        for (size_t k = 0; k < (size_t)devices; ++k) used[k] = !subs[k].empty();
        future_loads[d] += lens[s];
        used[d] = 1;
        int64_t max_seq = std::max(step_max, lens[s]);
        for (size_t added = 0; added <= (size_t)devices; ++added) {
            int64_t hi = 0, lo = 0;
            size_t nonempty = 0;
            for (size_t k = 0; k < (size_t)devices; ++k) {
                if (!used[k]) continue;
                if (nonempty == 0) {
                    hi = lo = future_loads[k];
                } else {
                    hi = std::max(hi, future_loads[k]);
                    lo = std::min(lo, future_loads[k]);
                }
                ++nonempty;
            }
            if (nonempty < 2) return true;
            if (hi - lo < max_seq) return true;
            if (hi - lo > max_seq) return false;
            if (added == (size_t)devices || next + added >= run.size()) return false;
            const int64_t l = lens[run[next + added]];
            size_t best = kNone;
            for (size_t k = 0; k < (size_t)devices; ++k) {
                if (future_loads[k] + l > budget) continue;
                if (best == kNone || future_loads[k] < future_loads[best]) best = k;
            }
            if (best == kNone) return false;
            future_loads[best] += l;
            used[best] = 1;
            max_seq = std::max(max_seq, l);
        }
        return false;
    }

    // Whether adding `s` to device `d` keeps the step's sub-batches within one maximal sequence of
    // each other in tokens. The spread is over the sub-batches the step hands out, so a step that
    // would hand out only one of them has nothing to bound, and a step whose sequences are all of
    // no tokens has no sequence to bound it by. A step that lands on the bound is admitted only
    // when the rest of the run can raise its bottom, and only within one length class.
    bool keeps_spread(size_t s, size_t d, int cls, const std::vector<size_t>& run,
                      size_t next) const {
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
        if (max_seq == 0) return true;
        const int64_t spread = hi - lo;
        if (spread < max_seq) return true;
        if (spread > max_seq) return false;
        for (size_t k = 0; k < (size_t)devices; ++k)
            if (!subs[k].empty() && classes[k] != cls) return false;
        return rebalances(s, d, run, next);
    }

    size_t pick(size_t s, int cls, const std::vector<size_t>& run, size_t next) const {
        size_t best = kNone;
        for (size_t k = 0; k < (size_t)devices; ++k) {
            if (sealed[k]) continue;
            if (!subs[k].empty() && classes[k] != cls) continue;
            if (loads[k] + lens[s] > budget) continue;
            if (!keeps_spread(s, k, cls, run, next)) continue;
            if (best == kNone || loads[k] < loads[best]) best = k;
        }
        return best;
    }

    void add(size_t s, size_t d, int cls) {
        subs[d].push_back(s);
        loads[d] += lens[s];
        classes[d] = cls;
        step_max = std::max(step_max, lens[s]);
    }

    void pack_bucket(const std::vector<size_t>& run, bool has_next) {
        for (size_t i = 0; i < run.size(); ++i) {
            const size_t s = run[i];
            const int cls = (int)std::bit_width((uint64_t)lens[s]);
            size_t d = pick(s, cls, run, i + 1);
            if (d == kNone) {
                close();
                d = pick(s, cls, run, i + 1);  // every sub-batch is empty again
            }
            add(s, d, cls);
        }
        size_t nonempty = 0;
        for (const auto& sub : subs)
            if (!sub.empty()) ++nonempty;
        if (nonempty == 0) return;
        if (has_next && (int)nonempty < devices) {
            // Carry this short step into the next class: keep what it has, open what it lacks.
            for (size_t k = 0; k < (size_t)devices; ++k) sealed[k] = !subs[k].empty();
        } else {
            close();
        }
    }
};

// A run whose sequences are all one length is the fixed-length or packed case a token-budget
// batcher exists for, and its step sizes can be planned exactly instead of discovered greedily.
// Every step before the tail has to fill every device, so with one length its size is a multiple of
// `devices`; the tail may hand out fewer devices, split into equal sub-batches that each fit `cap`
// sequences. Greedy filling spends the budget but can leave a tail that needs an extra step (a
// run of 101 at four per device on three devices ends on five, which no balanced step can hand
// out), so the sizes are chosen here to spend the budget on the fewest steps that fit those rules.
PlannedSteps plan_equal_run(const std::vector<size_t>& run, int64_t len, int64_t budget,
                            int devices) {
    const int64_t n = (int64_t)run.size();
    const int64_t d = devices;
    const int64_t cap = std::min<int64_t>(budget / len, n);
    auto tail_size = [&](int64_t p, int64_t& q_out) {
        if (p <= 0) return false;
        for (int64_t q = d; q >= 1; --q)
            if (p % q == 0 && p / q <= cap) {
                q_out = q;
                return true;
            }
        return false;
    };

    int64_t steps = 0, tail = 0, tail_devices = 1;
    const int64_t full_capacity = d * cap;
    // A plan always exists: at worst every step before the tail holds `devices` sequences and the
    // tail holds the remainder, which is one to `devices` sequences and splits into that many
    // one-sequence sub-batches. So this search stops at that plan at the latest.
    int64_t s = std::max<int64_t>(1, (n + full_capacity - 1) / full_capacity);
    for (;; ++s) {
        if (s == 1) {
            int64_t q = 1;
            if (!tail_size(n, q)) continue;
            steps = 1;
            tail = n;
            tail_devices = q;
            break;
        }
        const int64_t full_steps = s - 1;
        const int64_t min_full = full_steps * d;
        const int64_t max_full = full_steps * full_capacity;
        bool found = false;
        for (int64_t q = d; q >= 1 && !found; --q) {
            const int64_t low = std::max<int64_t>(q, n - max_full);
            const int64_t high = std::min<int64_t>(q * cap, n - min_full);
            if (low > high) continue;
            const int64_t k_lo = std::max<int64_t>(1, (low + q - 1) / q);
            const int64_t k_hi = std::min<int64_t>(cap, high / q);
            if (k_lo > k_hi) continue;
            // Any `d` consecutive k hold a solution of q*k == n (mod d), and d = devices is small.
            for (int64_t k = k_hi; k >= k_lo && k >= k_hi - d; --k) {
                if ((q * k - n) % d != 0) continue;
                tail = q * k;
                tail_devices = q;
                found = true;
                break;
            }
        }
        if (!found) continue;
        steps = s;
        break;
    }

    PlannedSteps plan;
    size_t pos = 0;
    int64_t remaining = n - tail;
    for (int64_t i = 0; i < steps - 1; ++i) {
        const int64_t left = (steps - 1) - i;
        const int64_t min_rest = (left - 1) * d;
        int64_t size = std::min(full_capacity, remaining - min_rest);
        size = (size / d) * d;
        Subs step((size_t)d);
        for (int64_t j = 0; j < size; ++j)
            step[(size_t)(j % d)].push_back(run[pos + (size_t)j]);
        pos += (size_t)size;
        remaining -= size;
        plan.push_back(std::move(step));
    }
    Subs last((size_t)tail_devices);
    for (int64_t j = 0; j < tail; ++j)
        last[(size_t)(j % tail_devices)].push_back(run[pos + (size_t)j]);
    plan.push_back(std::move(last));
    return plan;
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
    // Adjacent buckets of one length class are one run to the packer: the class is what keeps a
    // sub-batch homogeneous, and the 32-sequence chunks were the sort's unit, not the step's. A
    // step can then keep filling across that seam instead of closing on it, which is what lets a
    // fixed-length run spend its budget rather than stopping every chunk boundary.
    std::vector<std::vector<size_t>> runs;
    runs.reserve(buckets.size());
    for (auto& bucket : buckets) {
        const bool same_class =
            !runs.empty() && !bucket.empty() &&
            std::bit_width((uint64_t)lens[runs.back().front()]) ==
                std::bit_width((uint64_t)lens[bucket.front()]);
        if (same_class) {
            runs.back().insert(runs.back().end(), bucket.begin(), bucket.end());
        } else {
            runs.push_back(std::move(bucket));
        }
    }
    // A run whose sequences are all one length is the fixed-length case, and planning its step
    // sizes exactly is what keeps it from ending on a tail no balanced step can hand out.
    const bool one_length =
        runs.size() == 1 && !runs[0].empty() && lens[runs[0][0]] > 0 &&
        std::all_of(runs[0].begin(), runs[0].end(), [&](size_t i) {
            return lens[i] == lens[runs[0][0]];
        });
    EpochPlan p;
    if (one_length) {
        p.steps =
            plan_equal_run(runs[0], lens[runs[0][0]], config_.token_budget, config_.devices);
    } else {
        // A step normally never crosses a run edge, so classes do not meet in one sub-batch. The
        // exception is a run that ends with fewer sequences than devices: those sequences carry
        // forward to fill the next run's step instead of leaving empty devices behind.
        StepPacker packer(lens, config_.token_budget, config_.devices);
        for (size_t i = 0; i < runs.size(); ++i)
            packer.pack_bucket(runs[i], i + 1 < runs.size());
        p.steps = std::move(packer.steps);
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
