// r97/live_adapter.h -- the trainer's writer for radiance's runtime LoRA slot.
//
// A live adapter is one adapter slot on an engine that was started with runtime LoRA enabled
// (Config::lora targets, rank and slots). It is opened from configuration, filled only from
// WireBuffers (include/r97/delta.h), and updated in place: an update copies the adapter into the
// slot the next step already names, it does not reload a file.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "r97/delta.h"

namespace rad {
class Engine;
}

namespace r97 {

// What the engine must have room for, and what the slot is called. `targets` are HF module leaf
// names (q_proj, down_proj, ...); `rank` is the adapter's maximum rank, which the engine's slot
// must be at least; `alpha` is the adapter's scaling constant. A WireBuffers site carries its own
// scale, so alpha is not applied to it.
struct LiveAdapterConfig {
    std::string              name;
    std::vector<std::string> targets;
    int64_t                  rank = 0;
    double                   alpha = 0.0;
};

// One live slot. The engine owns the model, the adapter owns only the slot's handle: it must not
// outlive the engine, and no request may be running while update() replaces the weights.
class LiveAdapter {
public:
    // Reserve and publish the slot at zero (the base model's output). Throws nothing; on failure
    // returns nullptr and puts a message in `err` when `err` is non-null. The engine must already
    // be started.
    static std::unique_ptr<LiveAdapter> open(rad::Engine& engine, const LiveAdapterConfig& config,
                                            std::string* err = nullptr);
    ~LiveAdapter();

    LiveAdapter(const LiveAdapter&) = delete;
    LiveAdapter& operator=(const LiveAdapter&) = delete;

    // Replace the slot's weights with `wire`. One copy of the packed factors, no file read. The
    // next step that names slot() sees the new weights. Returns 0 on success and a negative
    // radiance error code otherwise, with a message in `err` when `err` is non-null.
    int update(const WireBuffers& wire, std::string* err = nullptr);

    // The slot every sequence of a step names. Valid from open() onward.
    int32_t slot() const;
    int64_t rank() const;
    double alpha() const;
    const std::string& name() const;

private:
    struct Impl;
    explicit LiveAdapter(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace r97
