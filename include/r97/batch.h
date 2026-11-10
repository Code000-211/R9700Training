// r97/batch.h -- what one training batch holds.
//
// A Batch is a list of sequences straight from the batcher. `score[i] == 1` means token `i` is a
// target, predicted at position `i - 1`; `score[0]` is 0, since nothing predicts it.
#pragma once

#include <cstdint>
#include <vector>

namespace r97 {

struct TokenSeq {
    std::vector<int32_t> ids;     // the sequence's token ids
    std::vector<uint8_t> score;   // 1 where token i is a target predicted at position i - 1
};

using Batch = std::vector<TokenSeq>;

}  // namespace r97
