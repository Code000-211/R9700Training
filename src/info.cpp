// src/info.cpp -- the `info` subcommand's implementation: a container's architecture,
// quantisation, layer count and hidden size, read through rad_core. Reading metadata needs no
// card, so the model test runs wherever the container is.
#include "info.h"

#include "radfile.h"

#include <iostream>
#include <string>

namespace r97 {

// Exit codes the CLI and the tests rely on: 0 ok, 3 the container did not open.
static constexpr int kInfoOk = 0;
static constexpr int kInfoBadFile = 3;

int cmd_info(const std::string& path, std::ostream& out) {
    rad::RadFile f;
    if (f.open(path.c_str()) != RAD_OK)
        return kInfoBadFile;

    const RadFileHeader& h = f.header();
    // rad-convert writes arch and quant both into the header strings and into the metadata; the
    // metadata wins where present (later writes), the header is the fallback.
    const char* arch  = f.get_s("arch");
    const char* quant = f.get_s("quant");
    if (!arch)  arch  = f.str(h.arch_id);
    if (!quant) quant = f.str(h.quant);
    int64_t layers = 0, hidden = 0;
    f.get_i("n_layers", &layers);
    f.get_i("n_embd", &hidden);

    out << "arch:     " << arch  << "\n"
        << "quant:    " << quant << "\n"
        << "layers:   " << layers << "\n"
        << "hidden:   " << hidden << "\n";
    return kInfoOk;
}

}  // namespace r97
