// tests/support/gpu_guard.h -- the GPU test guard: card selection, a lock per card, free VRAM.
//
// Several people and several agents run GPU tests on one machine. The guard keeps them apart:
//
//   R97_DEVICE selects the card (default 0). A test's DeviceGuard holds an exclusive flock on
//   <R97_LOCK_DIR|/tmp>/r97-card<N>.lock for its whole lifetime, so two runs started together
//   queue instead of failing (and ctest's RUN_SERIAL keeps the tier binary from stepping on
//   itself: CMakeLists.txt sets that property).
//
//   The test declares the VRAM it needs in GuardOptions::need_vram. Before doing anything the
//   guard reads /sys/class/drm/<card>/device/mem_info_vram_{total,used} for the selected card and
//   refuses at once, naming the free and the needed amounts and the processes that hold the card
//   (through amd-smi), rather than letting the test die with an allocation error halfway.
//
//     r97test::GuardOptions o;
//     o.device   = r97test::parse_device(getenv("R97_DEVICE") ? getenv("R97_DEVICE") : "0");
//     o.bdf      = pci_bus_id_of_hip_device(o.device);   // support_bdf below
//     o.need_vram = int64_t(1) << 30;                    // this test needs 1 GiB
//     r97test::DeviceGuard guard(o);                     // locks, then checks; throws with detail
//
// The pure parts -- parsing, card resolution, holder lists, the refusal text -- are free
// functions over a sysfs root or a CSV string, so host tests (tests/host/gpu_guard.cpp) run the
// guard's logic against a fake filesystem with no card at all.
//
// Everything here is POSIX (flock, /sys, amd-smi); the host tier still runs anywhere because host
// tests exercise these functions, not the machine's own files.
#pragma once

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// kSkip and kMiB come from tests/support/model_dir.h.
#include "support/model_dir.h"

namespace r97test {

/* ------------------------------------------------------------------ card selection */

// R97_DEVICE's value: a non-negative card ordinal. Throws naming the variable otherwise.
inline int parse_device(const std::string& value) {
    size_t i = 0;
    while (i < value.size() && std::isspace((unsigned char)value[i]))
        ++i;
    size_t digits = 0;
    while (i < value.size() && std::isdigit((unsigned char)value[i]))
        ++i, ++digits;
    while (i < value.size() && std::isspace((unsigned char)value[i]))
        ++i;
    if (digits == 0 || i != value.size())
        throw std::runtime_error("R97_DEVICE='" + value + "' is not a card ordinal (non-negative "
                                 "integer); fix it or unset it for card 0");
    long long n = std::strtoll(value.c_str(), nullptr, 10);
    return (int)n;
}

/* ------------------------------------------------------------------ small file reads */

// A /sys number: the first long the file holds, whitespace trimmed. Throws when unreadable.
inline int64_t read_sys_int(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f)
        throw std::runtime_error("cannot read " + path + ": " + std::strerror(errno));
    long long v = -1;
    int got = std::fscanf(f, "%lld", &v);
    std::fclose(f);
    if (got != 1)
        throw std::runtime_error("no number in " + path);
    return v;
}

// First line of a text file, newline trimmed; "" when the file is missing.
inline std::string read_line(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f)
        return "";
    char buf[512];
    const char* got = std::fgets(buf, sizeof buf, f);
    std::fclose(f);
    if (!got)
        return "";
    std::string line(buf);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        line.pop_back();
    return line;
}

/* ------------------------------------------------------------------ sysfs cards */

struct CardVram {
    int64_t total = 0;  // bytes the card has (mem_info_vram_total)
    int64_t used = 0;   // bytes all processes hold (mem_info_vram_used)
};

// <sysfs_root>/<card>/device/mem_info_vram_{total,used}. Throws when either file is unreadable.
inline CardVram read_card_vram(const std::string& sysfs_root, const std::string& card) {
    CardVram v;
    v.total = read_sys_int(sysfs_root + "/" + card + "/device/mem_info_vram_total");
    v.used  = read_sys_int(sysfs_root + "/" + card + "/device/mem_info_vram_used");
    return v;
}

inline bool is_card_dir(const char* name) {
    if (name[0] != 'c' || name[1] != 'a' || name[2] != 'r' || name[3] != 'd')
        return false;
    for (const char* p = name + 4; *p; ++p)
        if (!std::isdigit((unsigned char)*p))
            return false;
    return name[4] != '\0';
}

// Every cardN directory under the root with a mem_info_vram_total, in ascending card number:
// /sys/class/drm also holds cardN-<connector> links with no VRAM of their own.
inline std::vector<std::string> list_card_dirs(const std::string& sysfs_root) {
    DIR* d = opendir(sysfs_root.c_str());
    if (!d)
        throw std::runtime_error("cannot list " + sysfs_root + ": " + std::strerror(errno));
    std::vector<std::string> cards;
    while (dirent* e = readdir(d)) {
        if (!is_card_dir(e->d_name))
            continue;
        std::string card = e->d_name;
        std::string probe = sysfs_root + "/" + card + "/device/mem_info_vram_total";
        struct stat st;
        if (stat(probe.c_str(), &st) == 0)
            cards.push_back(card);
    }
    closedir(d);
    std::sort(cards.begin(), cards.end(),
              [](const std::string& a, const std::string& b) {
                  // Both are "card" plus digits; longer number means larger number.
                  if (a.size() != b.size())
                      return a.size() < b.size();
                  return a < b;
              });
    return cards;
}

inline std::vector<std::string> list_cards(const std::string& sysfs_root) {
    return list_card_dirs(sysfs_root);
}

// The card's PCI slot, from its uevent's PCI_SLOT_NAME (any line of the file); "" when absent.
inline std::string card_pci_slot(const std::string& sysfs_root, const std::string& card) {
    std::string uevent;
    std::FILE* f = std::fopen((sysfs_root + "/" + card + "/device/uevent").c_str(), "r");
    if (!f)
        return "";
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0)
        uevent.append(buf, n);
    std::fclose(f);
    const std::string key = "PCI_SLOT_NAME=";
    size_t at = uevent.find(key);
    if (at == std::string::npos)
        return "";
    std::string slot = uevent.substr(at + key.size());
    size_t end = slot.find('\n');
    return end == std::string::npos ? slot : slot.substr(0, end);
}

inline bool ieq(const std::string& a, const std::string& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
            return false;
    return true;
}

// The sysfs card whose PCI slot is <bdf> (HIP prints "0000:43:00.0"), or "" when none matches.
inline std::string card_for_bdf(const std::string& sysfs_root, const std::string& bdf) {
    for (const std::string& card : list_card_dirs(sysfs_root))
        if (ieq(card_pci_slot(sysfs_root, card), bdf))
            return card;
    return "";
}

/* ------------------------------------------------------------------ holders */

struct Holder {
    int gpu = 0;
    int pid = 0;
    std::string name;   // process name, "" when unknown (amd-smi prints N/A until asked)
    int64_t vram = 0;   // bytes of vram_mem the process holds
};

// A row of `amd-smi process --csv`:
//     gpu,name,pid,gtt_mem,cpu_mem,vram_mem,mem_usage,gfx,enc,sdma_usage,cu_occupancy,evicted_time
// Header lines and rows without VRAM are dropped.
inline std::vector<Holder> parse_holders(const std::string& amd_smi_csv) {
    std::vector<Holder> out;
    size_t pos = 0;
    while (pos < amd_smi_csv.size()) {
        size_t end = amd_smi_csv.find('\n', pos);
        std::string line = amd_smi_csv.substr(pos, end == std::string::npos ? std::string::npos
                                                                            : end - pos);
        pos = end == std::string::npos ? amd_smi_csv.size() : end + 1;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        std::vector<std::string> fields;
        size_t p = 0;
        while (true) {
            size_t c = line.find(',', p);
            fields.push_back(line.substr(p, c == std::string::npos ? std::string::npos
                                                                   : c - p));
            if (c == std::string::npos)
                break;
            p = c + 1;
        }
        if (fields.size() < 6 || fields[2].empty() || !std::isdigit((unsigned char)fields[2][0]))
            continue;  // header or junk
        long long pid = std::strtoll(fields[2].c_str(), nullptr, 10);
        long long vram = std::strtoll(fields[5].c_str(), nullptr, 10);
        if (vram <= 0)
            continue;
        Holder h;
        h.gpu = (int)std::strtoll(fields[0].c_str(), nullptr, 10);
        h.pid = (int)pid;
        h.name = fields[1] == "N/A" ? "" : fields[1];
        h.vram = vram;
        out.push_back(h);
    }
    return out;
}

// The kernel's name for the process, /proc/<pid>/comm; "" when it is gone.
inline std::string process_name(int pid) {
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%d/comm", pid);
    return read_line(path);
}

/* ------------------------------------------------------------------ the refusal text */

// The message a short card produces. Names the card, the free and the needed amounts and who
// holds the card.
inline std::string vram_short_text(const std::string& card, int64_t free_bytes, int64_t need_bytes,
                                   const std::vector<Holder>& holders) {
    std::string text = card + ": free VRAM " + std::to_string(free_bytes / kMiB)
                       + " MiB, the test needs " + std::to_string(need_bytes / kMiB)
                       + " MiB.";
    if (holders.empty()) {
        text += " No holder listed: run amd-smi process to see what holds the card.";
        return text;
    }
    text += " Processes that hold the card:";
    for (const Holder& h : holders) {
        std::string name = h.name.empty() ? process_name(h.pid) : h.name;
        if (name.empty())
            name = "?";
        text += "\n  gpu " + std::to_string(h.gpu) + " pid " + std::to_string(h.pid) + " " + name
                + " " + std::to_string(h.vram / kMiB) + " MiB";
    }
    return text;
}

/* ------------------------------------------------------------------ the guard */

struct GuardOptions {
    int device = 0;          // the card ordinal, from R97_DEVICE
    int64_t need_vram = 0;   // bytes this test declares; 0 skips the VRAM check
    std::string bdf;         // the device's PCI bus id ("0000:43:00.0"); empty means fall back:
                             // one card on the machine is unambiguous, more than one is an error
    std::string sysfs_root = "/sys/class/drm";
    std::string lock_dir;    // empty: R97_LOCK_DIR, else /tmp
    std::string amd_smi = "amd-smi";  // the command to list holders with; empty skips listing
    bool fail_fast = false;  // do not queue: throw at once when another run holds the card
};

// Holds a flock on the selected card for its lifetime (blocking: runs started together queue),
// then checks free VRAM and throws at once with detail when the test cannot fit.
class DeviceGuard {
public:
    explicit DeviceGuard(const GuardOptions& o) {
        const std::string lock_dir =
            o.lock_dir.empty()
                ? std::string(std::getenv("R97_LOCK_DIR") ? std::getenv("R97_LOCK_DIR") : "/tmp")
                : o.lock_dir;
        mkdir(lock_dir.c_str(), 0777);  // exists is fine
        lock_path_ = lock_dir + "/r97-card" + std::to_string(o.device) + ".lock";
        fd_ = open(lock_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
        if (fd_ < 0)
            throw std::runtime_error("cannot open the card lock " + lock_path_ + ": "
                                     + std::strerror(errno));

        int op = LOCK_EX;
        if (o.fail_fast)
            op |= LOCK_NB;
        while (flock(fd_, op) != 0) {
            if (errno == EINTR)
                continue;
            int why = errno;
            close(fd_);
            fd_ = -1;
            if (o.fail_fast && why == EWOULDBLOCK)
                throw std::runtime_error("card " + std::to_string(o.device)
                                         + " is held by another run (lock " + lock_path_
                                         + "); fail_fast refuses to queue behind it");
            throw std::runtime_error(std::string("cannot flock ") + lock_path_ + ": "
                                     + std::strerror(why));
        }

        // A failed constructor destroys nothing, so the lock must be released here: a thrown
        // guard must not keep holding the card.
        try {
            resolve_card(o);
            check_vram(o);
        } catch (...) {
            close(fd_);
            fd_ = -1;
            throw;
        }
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    ~DeviceGuard() {
        if (fd_ >= 0) {
            flock(fd_, LOCK_UN);
            close(fd_);
        }
    }

    const std::string& card() const { return card_; }
    const std::string& lock_path() const { return lock_path_; }
    const std::vector<Holder>& holders() const { return holders_; }

private:
    void resolve_card(const GuardOptions& o) {
        if (!o.bdf.empty()) {
            card_ = card_for_bdf(o.sysfs_root, o.bdf);
            if (card_.empty())
                throw std::runtime_error("no /sys/class/drm card for PCI bus id " + o.bdf
                                         + " (device " + std::to_string(o.device)
                                         + "): the guard cannot check free VRAM");
            return;
        }
        std::vector<std::string> cards = list_cards(o.sysfs_root);
        if (cards.empty())
            return;  // no sysfs VRAM anywhere; the lock still serialises runs
        if (cards.size() == 1) {
            card_ = cards[0];
            return;
        }
        std::string all;
        for (const std::string& c : cards)
            all += (all.empty() ? "" : ", ") + c;
        throw std::runtime_error("the machine has " + std::to_string(cards.size())
                                 + " cards with VRAM (" + all
                                 + "); pass GuardOptions::bdf for device "
                                 + std::to_string(o.device)
                                 + " so the guard checks the right card");
    }

    std::vector<Holder> list_holders(const GuardOptions& o) {
        holders_available_ = !o.amd_smi.empty();
        if (!holders_available_)
            return {};
        std::string cmd = o.amd_smi + " process -g " + std::to_string(o.device)
                          + " --csv 2>/dev/null";
        std::FILE* p = popen(cmd.c_str(), "r");
        if (!p) {
            holders_available_ = false;
            return {};
        }
        std::string csv;
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, p)) > 0)
            csv.append(buf, n);
        int status = pclose(p);
        holders_available_ = status == 0;
        return parse_holders(csv);
    }

    void check_vram(const GuardOptions& o) {
        if (card_.empty() || o.need_vram <= 0)
            return;
        CardVram v = read_card_vram(o.sysfs_root, card_);
        int64_t free_bytes = v.total - v.used;
        if (free_bytes >= o.need_vram)
            return;
        holders_ = list_holders(o);
        std::string text = vram_short_text(card_, free_bytes, o.need_vram, holders_);
        if (!holders_available_)
            text += "\n  (amd-smi did not answer; listed holders may be incomplete)";
        throw std::runtime_error(text);
    }

    int fd_ = -1;
    std::string lock_path_;
    std::string card_;
    std::vector<Holder> holders_;
    bool holders_available_ = true;
};

/* ------------------------------------------------------------------ HIP bus id */

// The PCI bus id of a HIP device, HIP-style extern. gpu-tier tests need no ROCm include path for
// this; __has_include prefers the real header when the build reaches one.
#if defined(__has_include)
#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#endif
#endif
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
extern "C" int hipDeviceGetPCIBusId(char* pciBusId, int len, int device);
#endif

inline std::string hip_bdf(int device) {
    char bus[32] = {0};
    if (hipDeviceGetPCIBusId(bus, sizeof bus, device) != 0)
        return "";
    return bus;
}

}  // namespace r97test
