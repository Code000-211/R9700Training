// tests/host/batcher.cpp -- the Batcher's budget, coverage, determinism, resume and balance.
//
// The host test binary has one `main`, in tests/host/vector_tape.cpp (the skeleton's own host
// test); a second one cannot link. This issue may neither edit that file nor decide the
// repository's host-test entry point on its own, so this file is self-contained: it carries its
// own runner and arms it with `std::atexit`, so the checks run once the binary's `main` returns.
// A failed `assert` aborts the process and `ctest` sees the failure. Nothing outside this file is
// involved; when the repository agrees on a single host-test entry point, the runner below is the
// only part that has to change.

// The default build type is RelWithDebInfo, which defines NDEBUG and turns assert into a no-op.
// In a test the assertions are the test, so keep them live in this translation unit -- before any
// header can pull <cassert> in.
#undef NDEBUG

#include "r97/batcher.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
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
                assert(!s.ids.empty());
                // recover the index from the first token id (ids encode i * 100003 + t)
                size_t i = (size_t)(s.ids[0] / 100003);
                assert(tokens(s) == lens[i]);
                for (int64_t t : s.ids) assert(t / 100003 == (int64_t)i);
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

std::vector<std::vector<Batch>> whole_run(Batcher& b, int64_t epochs) {
    std::vector<std::vector<Batch>> all;
    for (int64_t e = 0; e < epochs; ++e) {
        b.seek({e, 0});  // each epoch is its own shuffle
        auto one = run_epoch(b);
        assert((int64_t)one.size() == b.steps_in_epoch(e));
        for (auto& s : one) all.push_back(std::move(s));
    }
    return all;
}

// The budget (per sub-batch), coverage, determinism and resume of one Batcher over varied
// lengths.
void check_budget_coverage_and_resume() {
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
    assert(b.skipped().size() == 1 && b.skipped()[0] == seqs.size() - 1);
    assert(b.skipped_tokens() == 9000);

    auto run = whole_run(b, 3);

    // The budget holds for every Batch, not for a step over all of its sub-batches: a step can
    // spend up to devices * budget tokens, and on this data the largest step does exceed a whole
    // budget, so a batcher that bounded the step instead would fail here. A step's sequences are
    // of similar length: all of them come from one length bucket, whose lengths are within a
    // factor of two.
    int64_t largest_step_total = 0;
    for (const auto& step : run) {
        assert((int64_t)step.size() == devices);
        int64_t longest = 0, shortest = INT64_MAX, step_total = 0;
        for (const Batch& sub : step) {
            int64_t total = 0;
            for (const TokenSeq& s : sub) {
                total += tokens(s);
                longest = std::max(longest, tokens(s));
                shortest = std::min(shortest, tokens(s));
            }
            assert(total <= budget);
            step_total += total;
        }
        largest_step_total = std::max(largest_step_total, step_total);
        assert(shortest > 0 && longest < 2 * shortest);
    }
    assert(largest_step_total > budget);

    // Every kept sequence appears exactly once per epoch, and no two epochs are the same shuffle.
    std::vector<std::vector<size_t>> orders;
    {
        size_t lo = 0;
        for (int64_t e = 0; e < 3; ++e) {
            size_t hi = lo + (size_t)b.steps_in_epoch(e);
            std::vector<std::vector<Batch>> epoch(run.begin() + (long)lo, run.begin() + (long)hi);
            auto seen = coverage(epoch, lens);
            std::set<size_t> uniq(seen.begin(), seen.end());
            assert(seen.size() == 300 && uniq.size() == 300);
            assert(uniq.find(300) == uniq.end());  // the skipped one is not in any batch
            orders.push_back(std::move(seen));
            lo = hi;
        }
        assert(orders[0] != orders[1] && orders[1] != orders[2]);
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
    assert(differs);
    Batcher b3(seqs, cfg);
    auto run_again = whole_run(b3, 3);
    assert(run_again.size() == run.size());
    for (size_t k = 0; k < run.size(); ++k) {
        assert(run_again[k].size() == run[k].size());
        for (size_t d = 0; d < run[k].size(); ++d) {
            assert(run_again[k][d].size() == run[k][d].size());
            for (size_t i = 0; i < run[k][d].size(); ++i)
                assert(run_again[k][d][i].ids == run[k][d][i].ids);
        }
    }

    // Restore gives the same continuation.
    Batcher keep(seqs, cfg);
    auto full = whole_run(keep, 2);
    Batcher part(seqs, cfg);
    r97::Batcher::Position pos{1, 2};
    part.seek(pos);                       // epoch 1, from step 2 on
    assert(part.position().epoch == 1 && part.position().step == 2);
    std::vector<std::vector<Batch>> resumed;
    std::vector<Batch> step;
    while (part.next(&step)) resumed.push_back(step);
    size_t offset = (size_t)keep.steps_in_epoch(0) + 2;
    assert(resumed.size() == full.size() - (size_t)keep.steps_in_epoch(0) - 2);
    for (size_t k = 0; k < resumed.size(); ++k) {
        assert(resumed[k].size() == full[offset + k].size());
        for (size_t d = 0; d < resumed[k].size(); ++d)
            for (size_t i = 0; i < resumed[k][d].size(); ++i)
                assert(resumed[k][d][i].ids == full[offset + k][d][i].ids);
    }

    // N sub-batches of a step differ by less than one maximal sequence in tokens.
    for (const auto& stp : run) {
        assert((int64_t)stp.size() == devices);
        std::vector<int64_t> loads(devices, 0);
        int64_t max_seq = 0;
        for (size_t d = 0; d < stp.size(); ++d)
            for (const TokenSeq& s : stp[d]) {
                loads[d] += tokens(s);
                max_seq = std::max(max_seq, tokens(s));
            }
        int64_t spread = *std::max_element(loads.begin(), loads.end()) -
                         *std::min_element(loads.begin(), loads.end());
        assert(spread < max_seq);
    }
}

// The budget is per sub-batch, not per step: four sequences, each too long to share a sub-batch
// with any other under the budget, still fill both devices of every step. A per-step budget would
// put one sequence in each step with the other device idle, the case a long-sequence run hits.
void check_budget_is_per_batch() {
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
    assert(run.size() == 2);  // no two of these fit one sub-batch together

    std::set<size_t> seen;
    for (const auto& step : run) {
        assert((int64_t)step.size() == devices);
        std::vector<int64_t> loads(devices, 0);
        int64_t max_seq = 0, step_total = 0;
        for (size_t d = 0; d < step.size(); ++d) {
            assert(!step[d].empty());  // every device works on every step
            for (const TokenSeq& s : step[d]) {
                loads[d] += tokens(s);
                max_seq = std::max(max_seq, tokens(s));
                seen.insert((size_t)(s.ids[0] / 100003));
            }
            assert(loads[d] <= budget);  // the budget belongs to each sub-batch
            step_total += loads[d];
        }
        assert(step_total > budget);  // a step may spend more than one sub-batch's budget
        int64_t spread = *std::max_element(loads.begin(), loads.end()) -
                         *std::min_element(loads.begin(), loads.end());
        assert(spread < max_seq);
    }
    assert(seen.size() == lens.size());  // every sequence appears once
}

// Grouping similar lengths may not collapse the epoch's order into one fixed sort: a dataset of a
// few dozen sequences, every bucket well under the cap, must still change its batches with the
// seed and with the epoch.
void check_small_dataset_shuffles() {
    for (const size_t n : {size_t(20), size_t(32)}) {
        std::vector<int64_t> lens;  // distinct lengths, none over the budget
        for (size_t i = 0; i < n; ++i) lens.push_back(17 + (int64_t)i * 3);
        auto seqs = make_sequences(lens);

        r97::BatcherConfig cfg;
        cfg.token_budget = 200;
        cfg.devices = 2;
        cfg.seed = 1;
        Batcher b(seqs, cfg);

        // Every epoch holds every sequence once, and each epoch is a new shuffle.
        std::vector<std::vector<std::vector<int32_t>>> epochs;
        for (int64_t e = 0; e < 3; ++e) {
            b.seek({e, 0});
            epochs.push_back(id_order(run_epoch(b)));
            assert(epochs.back().size() == n);
            std::set<int64_t> uniq;
            for (const auto& ids : epochs.back()) uniq.insert((int64_t)ids[0]);
            assert(uniq.size() == n);  // every sequence once, none twice
        }
        assert(epochs[0] != epochs[1] && epochs[1] != epochs[2]);

        // The same seed replays the same batches.
        Batcher replay(seqs, cfg);
        replay.seek({0, 0});
        assert(id_order(run_epoch(replay)) == epochs[0]);

        // Another seed does not.
        cfg.seed = 2;
        Batcher other(seqs, cfg);
        other.seek({0, 0});
        assert(id_order(run_epoch(other)) != epochs[0]);
    }
}

// Grouping: a batch does not mix one long sequence with many short ones. Two length classes 40x
// apart with a budget that would happily hold one long sequence beside dozens of shorts. The
// counts matter: 32 longs are exactly one bucket by themselves, so they cannot expose a step that
// crosses a bucket edge; 33 longs leave a bucket holding one long and the shorts that follow it in
// the sorted order, which is the case that mixes classes unless the grouping cuts at the length
// class and the packing stops at the bucket edge. Both counts are checked, over several epochs.
void check_similar_lengths_are_grouped() {
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
                assert(step.size() == 1);
                assert(!step[0].empty());
                ++steps;
                int64_t longest = 0, shortest = kLong;
                for (const TokenSeq& s : step[0]) {
                    longest = std::max(longest, tokens(s));
                    shortest = std::min(shortest, tokens(s));
                    ++seen;
                }
                // No step mixes the two classes: a batch is never one long and a crowd of shorts.
                assert(longest == shortest);
                if (longest == kLong) {
                    assert(!saw_short_step);  // grouped descending: every long step comes first
                    saw_long_step = true;
                } else {
                    assert(longest == kShort);
                    saw_short_step = true;
                }
            }
            assert(saw_long_step && saw_short_step);
            assert(seen == kLongs + kShorts);
            // Pairs of longs, the odd long alone when there is one, then one step of shorts.
            assert(steps == (kLongs + 1) / 2 + 1);
        }
    }
}

// A step whose whole set fits one step: it is one step, split across all devices, every device
// gets a sub-batch, and the spread stays below the maximal sequence. All lengths share one length
// class, so grouping puts them in one bucket and the whole set can indeed be one step.
void check_single_step_split() {
    const std::vector<int64_t> lens = {4, 4, 5, 5, 6, 6, 7, 7};
    auto seqs = make_sequences(lens);

    r97::BatcherConfig cfg;
    cfg.token_budget = 50;  // the whole set fits one step
    cfg.devices = 3;
    cfg.seed = 99;
    Batcher b(seqs, cfg);

    std::vector<Batch> step;
    assert(b.next(&step));
    assert(step.size() == 3);

    std::vector<int64_t> loads(3, 0);
    size_t seen = 0;
    int64_t max_seq = 0;
    for (const Batch& sub : step) {
        size_t d = (size_t)(&sub - step.data());
        assert(!sub.empty());
        for (const TokenSeq& s : sub) {
            loads[d] += tokens(s);
            max_seq = std::max(max_seq, tokens(s));
            ++seen;
        }
    }
    assert(seen == lens.size());
    int64_t spread = *std::max_element(loads.begin(), loads.end()) -
                     *std::min_element(loads.begin(), loads.end());
    assert(spread < max_seq);
    assert(!b.next(&step));  // one step holds the whole set
}

// reset() is the way back to the first epoch, and the constructor rejects a configuration that
// could never plan an epoch: a budget of no tokens and a step with no devices.
void check_reset_and_config_errors() {
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
    assert(!rejects(cfg));  // a workable configuration is accepted
    r97::BatcherConfig bad = cfg;
    bad.token_budget = 0;
    assert(rejects(bad));
    bad.token_budget = -1;
    assert(rejects(bad));
    bad = cfg;
    bad.devices = 0;
    assert(rejects(bad));
    bad.devices = -1;
    assert(rejects(bad));

    // One epoch from (0, 0), then the same epoch again after seeking elsewhere and resetting.
    Batcher b(seqs, cfg);
    auto first = run_epoch(b);
    assert(!first.empty());
    assert(b.position().epoch == 0 && b.position().step == (int64_t)first.size());

    b.seek({1, 0});
    assert(!run_epoch(b).empty());  // a different epoch, from a different shuffle
    b.reset();
    assert(b.position().epoch == 0 && b.position().step == 0);
    auto again = run_epoch(b);
    assert(again.size() == first.size());
    assert(id_order(again) == id_order(first));  // same seed, same batches

    // reset() also revives a finished traversal: the epoch's end consumes nothing, so the
    // position still points there and the next epoch needs a seek; reset() is the way back.
    std::vector<Batch> step;
    assert(!b.next(&step));
    assert(b.position().epoch == 0 && b.position().step == (int64_t)again.size());
    b.reset();
    assert(b.next(&step));
    assert(b.position().epoch == 0 && b.position().step == 1);
    assert(step.size() == again[0].size());
    for (size_t d = 0; d < step.size(); ++d)
        for (size_t i = 0; i < step[d].size(); ++i)
            assert(step[d][i].ids == again[0][d][i].ids);
}

void run_checks() {
    check_budget_coverage_and_resume();
    check_budget_is_per_batch();
    check_small_dataset_shuffles();
    check_similar_lengths_are_grouped();
    check_single_step_split();
    check_reset_and_config_errors();
    std::puts("batcher ok");
}

// Armed at static-initialisation time; runs when the binary's `main` returns.
struct ArmChecks {
    ArmChecks() { std::atexit(&run_checks); }
} arm_checks;

}  // namespace
