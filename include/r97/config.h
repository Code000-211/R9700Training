// r97/config.h -- one field per configuration key.
//
// Defaults live here and nowhere else. The issue that owns a key (the optimiser, the batcher,
// the training loop) sets its default and its valid range; docs/USAGE.md documents them.
#pragma once

#include <cstdint>
#include <string>

#include "r97/delta.h"

namespace r97 {

struct Config {
    // Paths. Input and output.
    std::string model;     // the .rad container to train against
    std::string data;      // the training sequences
    std::string heldout;   // the held-out sequences
    std::string out;       // where adapters and checkpoints go

    std::string objective = "ce";  // the loss's name

    DeltaConfig delta{.kind = "lora"};   // what is trained

    // The optimiser and the loop.
    double   lr = 0.0;
    double   weight_decay = 0.0;
    int64_t  warmup_steps = 0;
    int64_t  token_budget = 0;
    int64_t  accumulation = 0;
    int64_t  steps = 0;
    int64_t  epochs = 0;
    int64_t  seed = 0;
    int64_t  eval_every = 0;
    int64_t  checkpoint_every = 0;

    std::string base_id;   // the adapter trained from resumes from this checkpoint's id
};

}  // namespace r97
