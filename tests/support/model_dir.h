// tests/support/model_dir.h -- where the container lives, and what to do when it does not.
//
// Tests that open the real model container (the label `model`) resolve it through this header:
//
//   auto rad = r97test::resolve_container(&why);
//   if (!rad) { std::printf("%s\n", why.c_str()); return r97test::kSkip; }  // clear skip
//
// Resolution order:
//   1. $R97_MODEL_DIR when set: the variable may name a .rad file or a directory that holds the
//      container (the first *.rad inside, sorted; a matching qwen3.8-27b-iq4r-s-v4-rad name wins).
//   2. the R97_MODEL_RAD macro the build resolved at configure time, when that file exists,
//   3. the default directory /srv/models/qwen3.8-27b-iqr-v4.
// Anything else skips the test: the message says what it looked for and how to fetch the file.
#pragma once

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace r97test {

// A skipped test's exit status (autotools convention; ctest records the run, not a failure) and
// the MiB unit; shared with tests/support/gpu_guard.h, which includes this header.
inline constexpr int kSkip = 77;

inline constexpr int64_t kMiB = 1024 * 1024;

inline bool is_file(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

inline bool is_dir(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Every *.rad directly inside a directory, sorted. Only files; no recursion into fetch dirs.
inline std::vector<std::string> rad_files_in(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d)
        return out;
    while (dirent* e = readdir(d)) {
        std::string name = e->d_name;
        if (name.size() < 5 || name.substr(name.size() - 4) != ".rad")
            continue;
        std::string path = dir + "/" + name;
        if (is_file(path))
            out.push_back(path);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    // The container this repository targets, before any other .rad lying in the same directory.
    std::string want = dir + "/qwen3.8-27b-iq4r-s-v4.rad";
    for (size_t i = 0; i < out.size(); ++i)
        if (out[i] == want)
            return {want};
    return out;
}

inline const char* default_model_dir() {
    return "/srv/models/qwen3.8-27b-iqr-v4";
}

// Resolves the container to a .rad path, or nothing. `why`, when given, takes the skip message.
// `explicit_rad`: a path the build already resolved (the R97_MODEL_RAD macro); used as step 2.
inline std::optional<std::string> resolve_container(const std::string& explicit_rad = "",
                                                    std::string* why = nullptr) {
    const char* env_dir = std::getenv("R97_MODEL_DIR");
    std::string dir = env_dir ? env_dir : default_model_dir();
    std::string looked;

    if (env_dir) {
        // The variable wins: it may name the .rad itself or a directory that holds it.
        if (is_file(dir))
            return dir;
        std::vector<std::string> rads = is_dir(dir) ? rad_files_in(dir) : std::vector<std::string>();
        if (!rads.empty())
            return rads[0];
        looked = dir + (is_dir(dir) ? " (no *.rad inside)" : " (no such directory)");
    } else {
        if (is_file(explicit_rad))
            return explicit_rad;
        looked = explicit_rad.empty() ? default_model_dir() : explicit_rad + ", " + default_model_dir();
        std::vector<std::string> rads = is_dir(default_model_dir())
                                            ? rad_files_in(default_model_dir())
                                            : std::vector<std::string>();
        if (!rads.empty())
            return rads[0];
    }

    if (why)
        *why = "skip: no model container. Looked: " + looked
               + ". Set R97_MODEL_DIR to the .rad or its directory, or fetch it:\n"
                 "  hf download StillDeadcode/qwen3.8-27b-iqr-v4 qwen3.8-27b-iq4r-s-v4.rad";
    return std::nullopt;
}

}  // namespace r97test
