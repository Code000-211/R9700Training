// tests/support/late_fixture.h -- the tiny fixture's container for checks that run at exit.
//
// The host tier's checks run from an atexit handler (tests/support/test.h), armed by the first
// check's static initialiser -- and a test object initialises before the library, so the handler
// runs after every other static in the binary has been destroyed. fixture() reads its directory
// from function-local statics (radtest::tmpdir(), detail::build_dir(), the fixture itself), which
// by then are freed memory: converting into that directory fails, or reads a stale pointer and
// crashes.
//
// The three paths the conversion needs are copied here while static initialisers still run, and the
// container is made when a check first asks for it, at exit. rad-convert is a separate process, so
// nothing of radiance's has to be alive for it; the checkpoint is written with the same
// write_checkpoint() the fixture uses.
#pragma once

#include "fixture.h"   // write_checkpoint, detail::rad_convert, detail::radiance_home

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace r97test {

// The paths, copied once while static initialisers run and kept (leaked on purpose: a check reads
// them at exit, after this binary's statics are gone).
struct LatePaths {
    std::string tmpdir;
    std::string rad_convert;
    std::string home;
};

// A directory of this process's own, under $RAD_TEST_TMPDIR, else $TMPDIR, else /tmp, as
// radtest::tmpdir() makes one -- made here because that helper's own string is one of the
// statics this file exists to avoid reading at exit.
inline std::string make_run_dir() {
    const char* base = std::getenv("RAD_TEST_TMPDIR");
    if (!base || !*base) base = std::getenv("TMPDIR");
    if (!base || !*base) base = "/tmp";
    std::string path = std::string(base) + "/r97late-XXXXXX";
    if (!::mkdtemp(path.data()))
        throw std::runtime_error(std::string("late_fixture: cannot make a directory under ") +
                                 base);
    return path;
}

inline LatePaths& late_paths() {
    static LatePaths* const paths = [] {
        LatePaths* p = new LatePaths;
        p->tmpdir      = make_run_dir();
        p->rad_convert = ::r97::detail::rad_convert();
        p->home        = ::r97::detail::radiance_home();
        return p;
    }();
    return *paths;
}

// The tiny fixture's container, converted on first use. Each call site includes one namespace-scope
// reference to late_paths() so that the copying above happens before the atexit runner, not inside
// it.
inline const std::string& late_container() {
    static const std::string container = [] {
        const LatePaths& p = late_paths();
        const std::string dir = p.tmpdir + "/tiny_qwen35_bf16";
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        ::r97::write_checkpoint(dir);
        const std::string out = dir + "/tiny_qwen35_bf16.rad";

        // The host kernel hierarchy is named explicitly so that the conversion and the engine
        // agree on it, as the fixture's own conversion does.
        std::string cmd = ::r97::detail::shell_quote(p.rad_convert) + " " +
                          ::r97::detail::shell_quote(dir) + " -o " +
                          ::r97::detail::shell_quote(out) + " --kernels libavx,libref";
        if (p.home.empty())
            throw std::runtime_error(
                "late_fixture: no plugin home: set RADIANCE_HOME to where radiance_home is");
        cmd += " --home " + ::r97::detail::shell_quote(p.home);
        if (std::system(cmd.c_str()) != 0)
            throw std::runtime_error("late_fixture: rad-convert failed: " + cmd);
        if (!std::filesystem::exists(out, ec))
            throw std::runtime_error("late_fixture: rad-convert wrote no container at " + out);
        return out;
    }();
    return container;
}

}  // namespace r97test
