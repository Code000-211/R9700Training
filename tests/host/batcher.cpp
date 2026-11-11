// tests/host/batcher.cpp -- the Batcher's budget, coverage, determinism, resume and balance.
//
// The checks register themselves with `R97_TEST` and fail through `R97_CHECK` (tests/support/
// test.h), so this file brings no `main` and no runner of its own: the host binary runs every
// registered test once its `main` returns.
#include "support/test.h"

#include "r97/batcher.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <vector>

using r97::Batch;
using r97::Batcher;
using r97::TokenSeq;

namespace {

int64_t tokens(const TokenSeq& s) { return (int64_t)s.ids.size(); }

// Deterministic pseudo-lengths so the expectations below never flake.
std::vector<int64_t> lengths_for(size_t n, uint64_t seed) {
    std::vector<int64_t> lens;
    lens.reserve(n);
    uint64_t state = seed * 6364136223846793005ull + 1442695040888963407ull;
    auto next = [&state] {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        return state;
    };
    for (size_t i = 0; i < n; ++i) lens.push_back(16 + (int64_t)(next() % 960));
    return lens;
}

std::vector<TokenSeq> make_sequences(const std::vector<int64_t>& lens) {
    std::vector<TokenSeq> out;
    out.reserve(lens.size());
    for (size_t i = 0; i < lens.size(); ++i) {
        TokenSeq s;
        s.ids.resize((size_t)lens[i]);
        for (int64_t t = 0; t < lens[i]; ++t) s.ids[(size_t)t] = (int32_t)(i * 100003 + t);
        s.score.assign(s.ids.size(), 0);
        out.push_back(std::move(s));
    }
    return out;
}

// The batches of one epoch, flattened: every original index, in order seen.
std::vector<size_t> coverage(const std::vector<std::vector<Batch>>& epoch,
                             const std::vector<int64_t>& lens) {
    std::vector<size_t> seen;
    for (size_t step = 0; step < epoch.size(); ++step) {
        for (const Batch& b : epoch[step]) {
            for (const TokenSeq& s : b) {
                R97_CHECK(!s.ids.empty());
                // recover the index from the first token id (ids encode i * 100003 + t)
                size_t i = (size_t)(s.ids[0] / 100003);
                R97_CHECK(tokens(s) == lens[i]);
                for (int64_t t : s.ids) R97_CHECK(t / 100003 == (int64_t)i);
                seen.push_back(i);
            }
        }
    }
    return seen;
}

std::vector<std::vector<Batch>> run_epoch(Batcher& b) {
    std::vector<std::vector<Batch>> steps;
    std::vector<Batch> step;
    while (b.next(&step)) steps.push_back(step);
    return steps;
}

// One epoch flattened to its sequences' token ids, which say both which sequence it is and in
// which order the batches hand it out.
std::vector<std::vector<int32_t>> id_order(const std::vector<std::vector<Batch>>& epoch) {
    std::vector<std::vector<int32_t>> out;
    for (const auto& step : epoch)
        for (const Batch& sub : step)
            for (const TokenSeq& s : sub) out.push_back(s.ids);
    return out;
}

// One epoch's whole shape: its steps, each step's sub-batches, each sub-batch's sequences by
// token ids. Two runs are equal only when every batch hands out the same sequences in the same
// order, which is what a tape reads back.
using EpochShape = std::vector<std::vector<std::vector<std::vector<int32_t>>>>;

EpochShape shape(const std::vector<std::vector<Batch>>& epoch) {
    EpochShape out;
    for (const auto& step : epoch) {
        std::vector<std::vector<std::vector<int32_t>>> s;
        for (const Batch& sub : step) {
            std::vector<std::vector<int32_t>> b;
            for (const TokenSeq& seq : sub) b.push_back(seq.ids);
            s.push_back(std::move(b));
        }
        out.push_back(std::move(s));
    }
    return out;
}

// One step's contract: at most `devices` sub-batches, none of them empty, each under the budget,
// and -- when the step hands out two or more -- a token spread under one maximal sequence of the
// step.
void check_step(const std::vector<Batch>& step, int64_t budget, int devices) {
    R97_CHECK(!step.empty());
    R97_CHECK((int)step.size() <= devices);
    std::vector<int64_t> loads;
    int64_t max_seq = 0;
    for (const Batch& sub : step) {
        R97_CHECK(!sub.empty());
        int64_t total = 0;
        for (const TokenSeq& s : sub) {
            total += tokens(s);
            max_seq = std::max(max_seq, tokens(s));
        }
        R97_CHECK(total <= budget);
        loads.push_back(total);
    }
    if (loads.size() >= 2) {
        const int64_t spread = *std::max_element(loads.begin(), loads.end()) -
                               *std::min_element(loads.begin(), loads.end());
        R97_CHECK(spread < max_seq);
    }
}

std::vector<std::vector<Batch>> whole_run(Batcher& b, int64_t epochs) {
    std::vector<std::vector<Batch>> all;
    for (int64_t e = 0; e < epochs; ++e) {
        b.seek({e, 0});  // each epoch is its own shuffle
        auto one = run_epoch(b);
        R97_CHECK((int64_t)one.size() == b.steps_in_epoch(e));
        for (auto& s : one) all.push_back(std::move(s));
    }
    return all;
}

// The budget (per sub-batch), coverage, determinism and resume of one Batcher over varied
// lengths.
R97_TEST(batcher_budget_coverage_and_resume) {
    const int64_t budget = 4096;
    const int devices = 2;
    auto lens = lengths_for(300, 7);
    auto seqs = make_sequences(lens);
    // One sequence over the budget: reported and skipped.
    seqs.push_back(TokenSeq{std::vector<int32_t>(9000, 1), std::vector<uint8_t>(9000, 1)});
    lens.push_back(9000);

    r97::BatcherConfig cfg;
    cfg.token_budget = budget;
    cfg.devices = devices;
    cfg.seed = 1234;
    Batcher b(seqs, cfg);

    // Skipped: the over-budget sequence, by index, never batched.
    R97_CHECK(b.skipped().size() == 1 && b.skipped()[0] == seqs.size() - 1);
    R97_CHECK(b.skipped_tokens() == 9000);

    auto run = whole_run(b, 3);

    // The budget holds for every Batch, not for a step over all of its sub-batches: a step can
    // spend up to devices * budget tokens, and on this data the largest step does exceed a whole
    // budget, so a batcher that bounded the step instead would fail here. A sub-batch's sequences
    // are of similar length -- they all come from one length bucket, whose lengths are within a
    // factor of two. A step may hold more than one class: a short tail carried into the next
    // bucket fills an idle device without joining the sub-batch that was already there. A step's
    // sub-batches differ by less than one maximal sequence in tokens; a step of one sub-batch has
    // no pair to differ. Which steps hand out fewer than `devices` is the next test's business.
    int64_t largest_step_total = 0;
    for (const auto& step : run) {
        check_step(step, budget, devices);
        int64_t step_total = 0;
        for (const Batch& sub : step) {
            int64_t total = 0, longest = 0, shortest = INT64_MAX;
            for (const TokenSeq& s : sub) {
                total += tokens(s);
                longest = std::max(longest, tokens(s));
                shortest = std::min(shortest, tokens(s));
            }
            R97_CHECK(shortest > 0 && longest < 2 * shortest);
            step_total += total;
        }
        largest_step_total = std::max(largest_step_total, step_total);
    }
    R97_CHECK(largest_step_total > budget);

    // Every kept sequence appears exactly once per epoch, and no two epochs are the same shuffle.
    std::vector<std::vector<size_t>> orders;
    {
        size_t lo = 0;
        for (int64_t e = 0; e < 3; ++e) {
            size_t hi = lo + (size_t)b.steps_in_epoch(e);
            std::vector<std::vector<Batch>> epoch(run.begin() + (long)lo, run.begin() + (long)hi);
            auto seen = coverage(epoch, lens);
            std::set<size_t> uniq(seen.begin(), seen.end());
            R97_CHECK(seen.size() == 300 && uniq.size() == 300);
            R97_CHECK(uniq.find(300) == uniq.end());  // the skipped one is not in any batch
            orders.push_back(std::move(seen));
            lo = hi;
        }
        R97_CHECK(orders[0] != orders[1] && orders[1] != orders[2]);
    }

    // Same seed gives the same batches; another seed does not.
    r97::BatcherConfig other = cfg;
    other.seed = 4321;
    Batcher b2(seqs, other);
    auto run_other = whole_run(b2, 3);
    bool differs = false;
    for (size_t k = 0; k < std::min(run.size(), run_other.size()); ++k) {
        if (run[k].size() != run_other[k].size()) { differs = true; break; }
        for (size_t d = 0; d < run[k].size(); ++d)
            if (run[k][d].size() != run_other[k][d].size() ||
                run[k][d].empty() || run_other[k][d].empty() ||
                run[k][d][0].ids != run_other[k][d][0].ids) { differs = true; break; }
        if (differs) break;
    }
    R97_CHECK(differs);
    Batcher b3(seqs, cfg);
    auto run_again = whole_run(b3, 3);
    R97_CHECK(run_again.size() == run.size());
    for (size_t k = 0; k < run.size(); ++k) {
        R97_CHECK(run_again[k].size() == run[k].size());
        for (size_t d = 0; d < run[k].size(); ++d) {
            R97_CHECK(run_again[k][d].size() == run[k][d].size());
            for (size_t i = 0; i < run[k][d].size(); ++i)
                R97_CHECK(run_again[k][d][i].ids == run[k][d][i].ids);
        }
    }

    // Restore gives the same continuation.
    Batcher keep(seqs, cfg);
    auto full = whole_run(keep, 2);
    Batcher part(seqs, cfg);
    r97::Batcher::Position pos{1, 2};
    part.seek(pos);                       // epoch 1, from step 2 on
    R97_CHECK(part.position().epoch == 1 && part.position().step == 2);
    std::vector<std::vector<Batch>> resumed;
    std::vector<Batch> step;
    while (part.next(&step)) resumed.push_back(step);
    size_t offset = (size_t)keep.steps_in_epoch(0) + 2;
    R97_CHECK(resumed.size() == full.size() - (size_t)keep.steps_in_epoch(0) - 2);
    for (size_t k = 0; k < resumed.size(); ++k) {
        R97_CHECK(resumed[k].size() == full[offset + k].size());
        for (size_t d = 0; d < resumed[k].size(); ++d)
            for (size_t i = 0; i < resumed[k][d].size(); ++i)
                R97_CHECK(resumed[k][d][i].ids == full[offset + k][d][i].ids);
    }
}

// Every step hands out `devices` sub-batches wherever the epoch still has sequences to fill one:
// only the last tail, with fewer sequences left than devices, may be short. A mid-epoch short
// step leaves a device idle while there is data still to hand out, which is what packing one
// bucket at a time does at an odd residue -- three longs beside three shorts on two devices, or
// varied lengths on three, where a bucket ends one or two sequences short of a full step.
R97_TEST(batcher_fills_every_device_mid_epoch) {
    struct Case {
        std::vector<int64_t> lens;
        int64_t budget;
        int devices;
    };
    const std::vector<int64_t> varied = lengths_for(800, 7);
    const std::vector<Case> cases = {
        // Three longs beside three shorts: the third long has no room in the step that pairs the
        // first two and no bound admits it beside a short, so it opens the next step, and the
        // shorts then fill the device it left idle.
        {{500, 500, 500, 12, 12, 12}, 1000, 2},
        {varied, 4096, 2},
        {varied, 4096, 3},
        {varied, 4096, 4},
        // One length class whose count is not a multiple of the devices per step.
        {std::vector<int64_t>(101, 500), 2000, 3},
    };
    for (const Case& c : cases) {
        auto seqs = make_sequences(c.lens);
        for (uint64_t seed : {1ull, 2ull, 99ull}) {
            for (int64_t epoch = 0; epoch < 3; ++epoch) {
                r97::BatcherConfig cfg;
                cfg.token_budget = c.budget;
                cfg.devices = c.devices;
                cfg.seed = seed;
                Batcher b(seqs, cfg);
                b.seek({epoch, 0});
                auto run = run_epoch(b);
                R97_CHECK((int64_t)run.size() == b.steps_in_epoch(epoch));
                R97_CHECK(!run.empty());
                auto seen = coverage(run, c.lens);
                R97_CHECK(seen.size() == c.lens.size());
                for (const auto& stp : run) check_step(stp, c.budget, c.devices);
                // Everything but the epoch's last tail has a step after it to carry into, so it
                // has no excuse for an idle device.
                for (size_t i = 0; i + 1 < run.size(); ++i)
                    R97_CHECK((int)run[i].size() == c.devices);
            }
        }
    }
}

// The budget is per sub-batch, not per step: four sequences, each too long to share a sub-batch
// with any other under the budget, still fill both devices of every step. A per-step budget would
// put one sequence in each step with the other device idle, the case a long-sequence run hits.
R97_TEST(batcher_budget_is_per_batch) {
    const int64_t budget = 4096;
    const int devices = 2;
    const std::vector<int64_t> lens = {2800, 3000, 3500, 2200};
    auto seqs = make_sequences(lens);

    r97::BatcherConfig cfg;
    cfg.token_budget = budget;
    cfg.devices = devices;
    cfg.seed = 17;
    Batcher b(seqs, cfg);

    auto run = run_epoch(b);
    R97_CHECK(run.size() == 2);  // no two of these fit one sub-batch together

    std::set<size_t> seen;
    for (const auto& step : run) {
        R97_CHECK((int64_t)step.size() == devices);
        std::vector<int64_t> loads(devices, 0);
        int64_t max_seq = 0, step_total = 0;
        for (size_t d = 0; d < step.size(); ++d) {
            R97_CHECK(!step[d].empty());  // every device works on every step
            for (const TokenSeq& s : step[d]) {
                loads[d] += tokens(s);
                max_seq = std::max(max_seq, tokens(s));
                seen.insert((size_t)(s.ids[0] / 100003));
            }
            R97_CHECK(loads[d] <= budget);  // the budget belongs to each sub-batch
            step_total += loads[d];
        }
        R97_CHECK(step_total > budget);  // a step may spend more than one sub-batch's budget
        int64_t spread = *std::max_element(loads.begin(), loads.end()) -
                         *std::min_element(loads.begin(), loads.end());
        R97_CHECK(spread < max_seq);
    }
    R97_CHECK(seen.size() == lens.size());  // every sequence appears once
}

// The shuffle must reach the batches at a realistic budget, not only when the budget is small
// enough that the packing has to end a step early. Wherever a whole bucket fits one step -- a
// dataset small next to the budget, a narrow length range, sequences all of one length -- the
// step partition follows from the bucket edges alone; the shuffled order the packing places the
// sequences in is then the only thing the seed and the epoch can change, and they must change it.
// Each case is a shape the fixed-sort batcher handed out identically for every seed and epoch.
R97_TEST(batcher_shuffle_reaches_the_batches) {
    struct Case {
        const char* what;  // for a reader of this file; the checks below do not print it
        std::vector<int64_t> lens;
        int64_t budget;
        int devices;
    };
    std::vector<Case> cases;
    {
        Case c{"20 sequences of 17..74 tokens, one device", {}, 4096, 1};
        for (size_t i = 0; i < 20; ++i) c.lens.push_back(17 + (int64_t)i * 3);
        cases.push_back(std::move(c));
    }
    {
        Case c{"100 sequences of one length, one device", {}, 4096, 1};
        c.lens.assign(100, 100);
        cases.push_back(std::move(c));
    }
    {
        Case c{"1000 sequences of 50..80 tokens, two devices", {}, 4096, 2};
        for (size_t i = 0; i < 1000; ++i) c.lens.push_back(50 + (int64_t)((i * 7) % 31));
        cases.push_back(std::move(c));
    }

    for (const Case& c : cases) {
        (void)c.what;
        auto seqs = make_sequences(c.lens);
        r97::BatcherConfig cfg;
        cfg.token_budget = c.budget;
        cfg.devices = c.devices;
        cfg.seed = 1;
        Batcher b(seqs, cfg);

        std::vector<EpochShape> epochs;
        for (int64_t e = 0; e < 3; ++e) {
            b.seek({e, 0});
            auto one = run_epoch(b);
            // Every sequence once an epoch, every sub-batch under the budget, and every step
            // with two sub-batches under the strict token spread.
            size_t seen = 0;
            for (const auto& step : one) {
                R97_CHECK(!step.empty());
                R97_CHECK((int)step.size() <= c.devices);
                std::vector<int64_t> loads(step.size(), 0);
                int64_t max_seq = 0;
                for (size_t d = 0; d < step.size(); ++d) {
                    R97_CHECK(!step[d].empty());
                    for (const TokenSeq& s : step[d]) {
                        loads[d] += tokens(s);
                        max_seq = std::max(max_seq, tokens(s));
                        ++seen;
                    }
                    R97_CHECK(loads[d] <= c.budget);
                }
                if (step.size() >= 2) {
                    int64_t spread = *std::max_element(loads.begin(), loads.end()) -
                                     *std::min_element(loads.begin(), loads.end());
                    R97_CHECK(spread < max_seq);
                }
            }
            R97_CHECK(seen == c.lens.size());
            epochs.push_back(shape(one));
        }
        // Each epoch is a new shuffle.
        R97_CHECK(epochs[0] != epochs[1] && epochs[1] != epochs[2]);

        // The same seed replays the same batches.
        Batcher replay(seqs, cfg);
        replay.seek({0, 0});
        R97_CHECK(shape(run_epoch(replay)) == epochs[0]);

        // Another seed does not.
        cfg.seed = 2;
        Batcher other(seqs, cfg);
        other.seek({0, 0});
        R97_CHECK(shape(run_epoch(other)) != epochs[0]);
    }
}

// A dataset whose buckets each hold one sequence: the shuffle has to reorder the buckets
// themselves. Shuffling only inside a bucket cannot touch a bucket of one, so with the grouping
// fixed by length and the bucket order fixed by the sort, every seed and every epoch handed out
// the same batches. The two shapes below are the ones a shuffle-inside-buckets batcher could not
// change: two sequences in different length classes, and one sequence per class across eight
// classes.
R97_TEST(batcher_shuffle_reaches_single_sequence_buckets) {
    struct Case {
        std::vector<int64_t> lens;
        int64_t budget;
        int devices;
    };
    std::vector<Case> cases;
    cases.push_back(Case{{3000, 700}, 4096, 2});
    cases.push_back(Case{{100, 300, 700, 1500, 3000, 6000, 12000, 24000}, 30000, 1});

    for (const Case& c : cases) {
        auto seqs = make_sequences(c.lens);
        r97::BatcherConfig cfg;
        cfg.token_budget = c.budget;
        cfg.devices = c.devices;
        cfg.seed = 1;

        // Each epoch is a new shuffle.
        Batcher b(seqs, cfg);
        std::vector<EpochShape> epochs;
        for (int64_t e = 0; e < 3; ++e) {
            b.seek({e, 0});
            auto one = run_epoch(b);
            R97_CHECK(coverage(one, c.lens).size() == c.lens.size());
            epochs.push_back(shape(one));
        }
        R97_CHECK(epochs[0] != epochs[1] || epochs[1] != epochs[2]);

        // Another seed does not replay this shuffle. Three seeds rather than a pair: two buckets
        // have only two orders, so one pair may coincide by chance while the seed still reaches
        // the batches.
        std::vector<EpochShape> seeds;
        for (uint64_t seed : {1, 2, 3}) {
            cfg.seed = seed;
            Batcher s(seqs, cfg);
            seeds.push_back(shape(run_epoch(s)));
        }
        R97_CHECK(seeds[0] != seeds[1] || seeds[1] != seeds[2]);

        // Same seed, same batches.
        cfg.seed = 1;
        Batcher again(seqs, cfg);
        R97_CHECK(shape(run_epoch(again)) == epochs[0]);
    }
}

// A dataset that fits one length bucket has no bucket order to shuffle. Sorting the bucket by
// length alone would then leave the seed and the epoch with nothing to change, so the one
// bucket's sequences are shuffled instead, under the seed and rotated by the epoch. This is the
// case the earlier round covered for a dataset small next to the budget: every sequence the same
// length, one bucket after grouping.
R97_TEST(batcher_single_bucket_shuffle) {
    // Two sequences is the smallest case: there are only two orders, so the three epochs cannot
    // all differ, but every consecutive pair must. Three sequences and a full bucket cover the
    // larger ones.
    for (int64_t count : {2, 3, 20}) {
        std::vector<int64_t> lens(static_cast<size_t>(count), 100);
        auto seqs = make_sequences(lens);

        r97::BatcherConfig cfg;
        cfg.token_budget = 4096;
        cfg.devices = 1;
        cfg.seed = 1;
        Batcher b(seqs, cfg);

        std::vector<EpochShape> epochs;
        for (int64_t e = 0; e < 3; ++e) {
            b.seek({e, 0});
            epochs.push_back(shape(run_epoch(b)));
        }
        R97_CHECK(epochs[0] != epochs[1] && epochs[1] != epochs[2]);

        cfg.seed = 2;
        Batcher other(seqs, cfg);
        R97_CHECK(shape(run_epoch(other)) != epochs[0]);
    }
}

// Grouping: a batch does not mix one long sequence with many short ones. Two length classes 40x
// apart with a budget that would happily hold one long sequence beside dozens of shorts. The
// counts matter: 32 longs are exactly one bucket by themselves, so they cannot expose a step that
// crosses a bucket edge; 33 longs leave a bucket holding one long, which carries into whichever
// bucket the epoch's shuffle puts next, which is the case that mixes classes unless the grouping
// cuts at the length class and the packing stops at the bucket edge. Both counts are checked, over
// several epochs; the bucket order is the epoch's to choose, so no check here fixes which class
// comes first.
R97_TEST(batcher_similar_lengths_are_grouped) {
    const int64_t kLong = 500;
    const int64_t kShort = 12;
    const size_t kShorts = 32;
    for (const size_t kLongs : {size_t(32), size_t(33)}) {
        std::vector<int64_t> lens;
        lens.reserve(kLongs + kShorts);
        for (size_t i = 0; i < kLongs; ++i) lens.push_back(kLong);
        for (size_t i = 0; i < kShorts; ++i) lens.push_back(kShort);
        auto seqs = make_sequences(lens);

        r97::BatcherConfig cfg;
        cfg.token_budget = 1000;  // one long and 41 shorts would fit, were they ever in one step
        cfg.devices = 1;
        cfg.seed = 5;
        Batcher b(seqs, cfg);

        for (int64_t e = 0; e < 4; ++e) {
            b.seek({e, 0});
            bool saw_long_step = false, saw_short_step = false;
            size_t steps = 0, seen = 0;
            std::vector<Batch> step;
            while (b.next(&step)) {
                R97_CHECK(step.size() == 1);
                R97_CHECK(!step[0].empty());
                ++steps;
                int64_t longest = 0, shortest = kLong;
                for (const TokenSeq& s : step[0]) {
                    longest = std::max(longest, tokens(s));
                    shortest = std::min(shortest, tokens(s));
                    ++seen;
                }
                // No step mixes the two classes: a batch is never one long and a crowd of shorts.
                // The shuffled bucket order decides which class comes first, so neither order is
                // assumed.
                R97_CHECK(longest == shortest);
                if (longest == kLong)
                    saw_long_step = true;
                else
                    saw_short_step = true;
            }
            R97_CHECK(saw_long_step && saw_short_step);
            R97_CHECK(seen == kLongs + kShorts);
            // Pairs of longs, the odd long alone when there is one, then one step of shorts.
            R97_CHECK(steps == (kLongs + 1) / 2 + 1);
        }
    }
}

// A step whose whole set fits one step: it is one step, split across all devices, every device
// gets a sub-batch, and the spread stays below the maximal sequence. All lengths share one length
// class, so grouping puts them in one bucket and the whole set can indeed be one step.
R97_TEST(batcher_single_step_split) {
    const std::vector<int64_t> lens = {4, 4, 5, 5, 6, 6, 7, 7};
    auto seqs = make_sequences(lens);

    r97::BatcherConfig cfg;
    cfg.token_budget = 50;  // the whole set fits one step
    cfg.devices = 3;
    cfg.seed = 99;
    Batcher b(seqs, cfg);

    std::vector<Batch> step;
    R97_CHECK(b.next(&step));
    R97_CHECK(step.size() == 3);

    std::vector<int64_t> loads(3, 0);
    size_t seen = 0;
    int64_t max_seq = 0;
    for (const Batch& sub : step) {
        size_t d = (size_t)(&sub - step.data());
        R97_CHECK(!sub.empty());
        for (const TokenSeq& s : sub) {
            loads[d] += tokens(s);
            max_seq = std::max(max_seq, tokens(s));
            ++seen;
        }
    }
    R97_CHECK(seen == lens.size());
    int64_t spread = *std::max_element(loads.begin(), loads.end()) -
                     *std::min_element(loads.begin(), loads.end());
    R97_CHECK(spread < max_seq);
    R97_CHECK(!b.next(&step));  // one step holds the whole set
}

// reset() is the way back to the first epoch, and the constructor rejects a configuration that
// could never plan an epoch: a budget of no tokens and a step with no devices.
R97_TEST(batcher_reset_and_config_errors) {
    auto lens = lengths_for(60, 11);
    auto seqs = make_sequences(lens);

    r97::BatcherConfig cfg;
    cfg.token_budget = 700;
    cfg.devices = 2;
    cfg.seed = 2024;

    auto rejects = [&](const r97::BatcherConfig& c) {
        try {
            Batcher b(seqs, c);
            (void)b;
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    R97_CHECK(!rejects(cfg));  // a workable configuration is accepted
    r97::BatcherConfig bad = cfg;
    bad.token_budget = 0;
    R97_CHECK(rejects(bad));
    bad.token_budget = -1;
    R97_CHECK(rejects(bad));
    bad = cfg;
    bad.devices = 0;
    R97_CHECK(rejects(bad));
    bad.devices = -1;
    R97_CHECK(rejects(bad));

    // One epoch from (0, 0), then the same epoch again after seeking elsewhere and resetting.
    Batcher b(seqs, cfg);
    auto first = run_epoch(b);
    R97_CHECK(!first.empty());
    R97_CHECK(b.position().epoch == 0 && b.position().step == (int64_t)first.size());

    b.seek({1, 0});
    R97_CHECK(!run_epoch(b).empty());  // a different epoch, from a different shuffle
    b.reset();
    R97_CHECK(b.position().epoch == 0 && b.position().step == 0);
    auto again = run_epoch(b);
    R97_CHECK(again.size() == first.size());
    R97_CHECK(id_order(again) == id_order(first));  // same seed, same batches

    // reset() also revives a finished traversal: the epoch's end consumes nothing, so the
    // position still points there and the next epoch needs a seek; reset() is the way back.
    std::vector<Batch> step;
    R97_CHECK(!b.next(&step));
    R97_CHECK(b.position().epoch == 0 && b.position().step == (int64_t)again.size());
    b.reset();
    R97_CHECK(b.next(&step));
    R97_CHECK(b.position().epoch == 0 && b.position().step == 1);
    R97_CHECK(step.size() == again[0].size());
    for (size_t d = 0; d < step.size(); ++d)
        for (size_t i = 0; i < step[d].size(); ++i)
            R97_CHECK(step[d][i].ids == again[0][d][i].ids);
}

// A short tail of a length bucket -- fewer sequences than devices -- must not be handed out as
// an empty sub-batch. It carries into the next bucket to fill the idle devices; the last tail of
// an epoch, with nothing left to carry into, hands out fewer sub-batches rather than an empty
// one. Every step that still has two or more sub-batches keeps the strict balance: the token
// spread is less than the step's longest sequence. The cases below are the ones a per-bucket
// greedy gets wrong: a long bucket tail beside a short bucket, a final odd sequence, and an odd
// residue that greedy would hand to one device while leaving the other behind. The epoch's
// shuffled bucket order may put the one-long tail last, where it has nothing to carry into, so
// for that case a short step is allowed only as the final step.
R97_TEST(batcher_short_tail_balance) {
    struct Case {
        const char* what;
        std::vector<int64_t> lens;
        int64_t budget;
        int devices;
        bool carry_must_fill;  // every short step here would be a carry that failed
    };
    std::vector<Case> cases;
    {
        Case c{"33 longs of 500 beside 32 shorts of 12, two devices", {}, 1000, 2, true};
        c.lens.assign(33, 500);
        c.lens.insert(c.lens.end(), 32, 12);
        cases.push_back(std::move(c));
    }
    {
        Case c{"5 sequences of 500, budget 1200, two devices", {}, 1200, 2, false};
        c.lens.assign(5, 500);
        cases.push_back(std::move(c));
    }
    {
        Case c{"3 sequences of 500, budget 1200, two devices", {}, 1200, 2, false};
        c.lens.assign(3, 500);
        cases.push_back(std::move(c));
    }
    {
        Case c{"3 unit sequences, two devices", {}, 100, 2, false};
        c.lens.assign(3, 1);
        cases.push_back(std::move(c));
    }

    for (const Case& c : cases) {
        (void)c.what;
        auto seqs = make_sequences(c.lens);
        r97::BatcherConfig cfg;
        cfg.token_budget = c.budget;
        cfg.devices = c.devices;
        cfg.seed = 23;
        Batcher b(seqs, cfg);

        auto run = run_epoch(b);
        R97_CHECK(!run.empty());
        size_t seen = 0;
        bool saw_short = false;
        for (const auto& step : run) {
            R97_CHECK(!step.empty());
            R97_CHECK((int)step.size() <= c.devices);
            if ((int)step.size() < c.devices) saw_short = true;
            std::vector<int64_t> loads(step.size(), 0);
            int64_t max_seq = 0;
            for (size_t d = 0; d < step.size(); ++d) {
                R97_CHECK(!step[d].empty());  // never an empty sub-batch for a device
                for (const TokenSeq& s : step[d]) {
                    loads[d] += tokens(s);
                    max_seq = std::max(max_seq, tokens(s));
                    ++seen;
                }
                R97_CHECK(loads[d] <= c.budget);
            }
            if (step.size() >= 2) {
                int64_t spread = *std::max_element(loads.begin(), loads.end()) -
                                 *std::min_element(loads.begin(), loads.end());
                R97_CHECK(spread < max_seq);
            }
        }
        R97_CHECK(seen == c.lens.size());
        if (c.carry_must_fill) {
            for (size_t i = 0; i < run.size(); ++i)
                if ((int)run[i].size() < c.devices) R97_CHECK(i + 1 == run.size());
        } else {
            R97_CHECK(saw_short);  // an epoch's final tail has nothing to fill with
        }
    }
}

// A saved Position is plain data, and seek() promises no validity beyond what a previous
// position() returned. A stepped-back or corrupted epoch/step must not index a plan out of bounds:
// seek() clamps it into the valid range, so a negative step resumes at the first step, a negative
// epoch resumes at the first shuffle, and a step past the end leaves the traversal at the epoch's
// end rather than reading a step that is not there.
R97_TEST(batcher_position_is_range_checked) {
    auto lens = lengths_for(80, 31);
    auto seqs = make_sequences(lens);

    r97::BatcherConfig cfg;
    cfg.token_budget = 512;
    cfg.devices = 2;
    cfg.seed = 808;
    Batcher b(seqs, cfg);

    b.seek({0, 0});
    auto first = run_epoch(b);
    R97_CHECK(!first.empty());

    b.seek({0, -1});
    R97_CHECK(b.position().epoch == 0 && b.position().step == 0);
    R97_CHECK(id_order(run_epoch(b)) == id_order(first));  // the first step, not plan[-1]

    b.seek({0, -123456});
    R97_CHECK(b.position().step == 0);
    R97_CHECK(id_order(run_epoch(b)) == id_order(first));

    b.seek({0, 2});
    auto from_two = run_epoch(b);
    b.seek({-7, 2});
    R97_CHECK(b.position().epoch == 0 && b.position().step == 2);
    R97_CHECK(id_order(run_epoch(b)) == id_order(from_two));

    b.seek({-7, -7});
    R97_CHECK(b.position().epoch == 0 && b.position().step == 0);
    R97_CHECK(id_order(run_epoch(b)) == id_order(first));

    const int64_t end = b.steps_in_epoch(0);
    b.seek({0, end + 1000});
    R97_CHECK(b.position().step == end);
    std::vector<Batch> step;
    R97_CHECK(!b.next(&step));
    R97_CHECK(b.position().step == end);
}

}  // namespace
