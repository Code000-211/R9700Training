// tests/host/batcher.cpp -- the Batcher's budget, coverage, determinism, resume and balance.
#include "support/test_main.h"

#include "r97/batcher.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <set>
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
    size_t steps = (size_t)epoch[0].size();
    for (size_t step = 0; step < steps; ++step) {
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

std::vector<std::vector<Batch>> whole_run(Batcher& b, int64_t epochs) {
    std::vector<std::vector<Batch>> all;
    for (int64_t e = 0; e < epochs; ++e) {
        auto one = run_epoch(b);
        assert((int64_t)one.size() == b.steps_in_epoch(e));
        for (auto& s : one) all.push_back(std::move(s));
        b.reset();
    }
    return all;
}

}  // namespace

R97_TEST(batcher) {
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

    auto run = whole_run(b, 3);

    // The budget holds for every sub-batch of every step.
    for (const auto& step : run)
        for (const Batch& sub : step) {
            int64_t total = 0;
            for (const TokenSeq& s : sub) total += tokens(s);
            assert(total <= budget);
        }

    // Every kept sequence appears exactly once per epoch.
    {
        std::vector<std::vector<Batch>> epoch0(run.begin(), run.begin() + b.steps_in_epoch(0));
        auto seen = coverage(epoch0, lens);
        std::set<size_t> uniq(seen.begin(), seen.end());
        assert(seen.size() == 300 && uniq.size() == 300);
        assert(uniq.find(300) == uniq.end());  // the skipped one is not in any batch
    }

    // Same seed gives the same batches; another seed does not.
    r97::BatcherConfig other = cfg;
    other.seed = 4321;
    Batcher b2(seqs, other);
    auto run_other = whole_run(b2, 1);
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
    auto run_again = whole_run(b3, 1);
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
