// tests/host/gpu_guard.cpp -- the GPU test guard's logic against a fake filesystem, no card.
//
// The host tier runs anywhere, and several copies of it may run at once. Everything here is a
// directory of its own under TMPDIR (TempDir), and every DeviceGuard is given that directory as
// its sysfs root and as its lock directory: the tests never read /sys, never open
// /tmp/r97-card<N>.lock, and so never queue behind a real gpu run or behind each other.
#include "support/gpu_guard.h"
#include "support/model_dir.h"
#include "support/test.h"

#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using r97test::CardVram;
using r97test::GuardOptions;
using r97test::Holder;

// A directory of our own, removed when it leaves scope.
class TempDir {
public:
    TempDir() {
        const char* base = std::getenv("TMPDIR");
        std::string tmpl = std::string(base && *base ? base : "/tmp") + "/r97-guard-XXXXXX";
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        if (!mkdtemp(buf.data()))
            throw std::runtime_error("mkdtemp " + tmpl + ": " + std::strerror(errno));
        path_ = buf.data();
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);  // best effort: a leftover temp dir must not fail the test
    }

    const std::string& path() const { return path_; }

    // Writes a file under the directory, creating the parent directories first.
    void write(const std::string& rel, const std::string& text) const {
        const fs::path p = fs::path(path_) / rel;
        fs::create_directories(p.parent_path());
        std::ofstream(p) << text << "\n";
    }

private:
    std::string path_;
};

// What a call threw, or a failure when it did not throw at all.
template <class Fn>
std::string thrown(Fn fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    throw std::runtime_error("expected a throw, none came");
}

// A fake /sys/class/drm: card0 is a connector-only entry with no VRAM of its own, card1, card2
// and card10 hold VRAM (card10 proves the numeric order of list_cards). card1 is nearly full --
// 70 MiB free of 32640 MiB -- and card2 is empty.
void fake_drm(const TempDir& dir) {
    dir.write("card0/device/uevent", "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:03:00.0");
    dir.write("card1/device/mem_info_vram_total", "34225520640");
    dir.write("card1/device/mem_info_vram_used", "34151870464");
    dir.write("card1/device/uevent", "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:43:00.0");
    dir.write("card2/device/mem_info_vram_total", "34225520640");
    dir.write("card2/device/mem_info_vram_used", "0");
    dir.write("card2/device/uevent", "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:c3:00.0");
    dir.write("card10/device/mem_info_vram_total", "34225520640");
    dir.write("card10/device/mem_info_vram_used", "0");
    dir.write("card10/device/uevent", "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:d3:00.0");
}

// One card only: card2, empty. A machine with one card needs no bdf to be unambiguous.
void fake_drm_one_card(const TempDir& dir) {
    dir.write("card2/device/mem_info_vram_total", "34225520640");
    dir.write("card2/device/mem_info_vram_used", "0");
    dir.write("card2/device/uevent", "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:c3:00.0");
}

R97_TEST(device_ordinal) {
    R97_CHECK(r97test::parse_device("0") == 0);
    R97_CHECK(r97test::parse_device("3") == 3);
    R97_CHECK(r97test::parse_device(" 7 ") == 7);
    for (const char* bad : {"", "-1", "x", "1x", "one"}) {
        const std::string why = thrown([bad] { (void)r97test::parse_device(bad); });
        R97_CHECK(why.find("R97_DEVICE") != std::string::npos);
    }
}

R97_TEST(sysfs_cards) {
    const TempDir drm;
    fake_drm(drm);

    const CardVram v = r97test::read_card_vram(drm.path(), "card1");
    R97_CHECK(v.total == 34225520640LL);
    R97_CHECK(v.used == 34151870464LL);

    // Only cards with their own VRAM list: the connector-only card0 does not.
    const std::vector<std::string> cards = r97test::list_cards(drm.path());
    R97_CHECK((cards == std::vector<std::string>{"card1", "card2", "card10"}));
    // An empty tree has no cards; a tree that is not there is an error, named.
    const TempDir empty;
    R97_CHECK(r97test::list_cards(empty.path()).empty());
    const std::string missing =
        thrown([&] { (void)r97test::list_cards((fs::path(drm.path()) / "nope").string()); });
    R97_CHECK(missing.find("nope") != std::string::npos);

    const std::string why = thrown([&] { (void)r97test::read_card_vram(drm.path(), "card0"); });
    R97_CHECK(why.find("mem_info_vram_total") != std::string::npos);
}

R97_TEST(bdf_resolution) {
    const TempDir drm;
    fake_drm(drm);
    const TempDir locks;  // the fake lock dir: the guard locks before it resolves

    R97_CHECK(r97test::card_pci_slot(drm.path(), "card1") == "0000:43:00.0");
    R97_CHECK(r97test::card_for_bdf(drm.path(), "0000:43:00.0") == "card1");
    R97_CHECK(r97test::card_for_bdf(drm.path(), "0000:C3:00.0") == "card2");  // case-insensitive
    R97_CHECK(r97test::card_for_bdf(drm.path(), "0000:99:00.0").empty());

    // Two cards and no bdf: the guard names the problem instead of checking the wrong card.
    GuardOptions two;
    two.device = 1;
    two.need_vram = 0;
    two.sysfs_root = drm.path();
    two.lock_dir = locks.path();
    const std::string why = thrown([&] { r97test::DeviceGuard g(two); });
    R97_CHECK(why.find("GuardOptions::bdf") != std::string::npos);

    // A guard whose constructor threw released the lock it had taken: this one does not queue
    // behind it.
    const TempDir one;
    fake_drm_one_card(one);
    GuardOptions unambiguous = two;
    unambiguous.sysfs_root = one.path();
    unambiguous.fail_fast = true;
    r97test::DeviceGuard ok(unambiguous);
    R97_CHECK(ok.card() == "card2");
    R97_CHECK(ok.lock_path() == locks.path() + "/r97-card1.lock");
}

R97_TEST(no_card_is_a_failure) {
    // No /sys/class/drm card at all -- a machine without a GPU, or a host-only build. The gpu
    // tier must fail here, at once and by name, not skip and not pass on a host backend.
    const TempDir empty;
    const TempDir locks;
    GuardOptions o;
    o.device = 0;
    o.need_vram = int64_t(1) << 30;
    o.sysfs_root = empty.path();
    o.lock_dir = locks.path();
    o.amd_smi = "";
    const std::string why = thrown([&] { r97test::DeviceGuard g(o); });
    R97_CHECK(why.find(empty.path()) != std::string::npos);
    R97_CHECK(why.find("R97_DEVICE") != std::string::npos);

    // Even a guard that declares no VRAM cannot run: the card itself is missing.
    o.need_vram = 0;
    const std::string why2 = thrown([&] { r97test::DeviceGuard g(o); });
    R97_CHECK(why2.find("no card with VRAM") != std::string::npos);
}

// A stub `amd-smi process -g N --csv`, printing what amd-smi prints on a held card: one line per
// process, the per-process record nested in one quoted field.
void stub_amd_smi(const TempDir& dir) {
    dir.write("amd-smi",
              "#!/bin/sh\n"
              "cat <<'R97STUB'\n"
              "gpu,process_info\n"
              "0,\"{'name': 'radiance', 'pid': 999, 'memory_usage': {'gtt_mem': 0, 'cpu_mem': 0, "
              "'vram_mem': 33280000000}, 'mem_usage': 0, 'usage': {'gfx': 0, 'enc': 0}, "
              "'sdma_usage': 0, 'cu_occupancy': 0, 'evicted_time': 0}\"\n"
              "R97STUB");
    const fs::path script = fs::path(dir.path()) / "amd-smi";
    if (chmod(script.c_str(), 0755) != 0)
        throw std::runtime_error("chmod " + script.string() + ": " + std::strerror(errno));
}

R97_TEST(vram_refusal) {
    const TempDir drm;
    fake_drm(drm);  // card1: 70 MiB free
    const TempDir locks;
    const TempDir bin;
    stub_amd_smi(bin);

    GuardOptions o;
    o.device = 0;
    o.bdf = "0000:43:00.0";
    o.need_vram = int64_t(1) << 30;  // 1 GiB against 70 MiB free
    o.sysfs_root = drm.path();
    o.lock_dir = locks.path();
    o.amd_smi = (fs::path(bin.path()) / "amd-smi").string();

    const std::string why = thrown([&] { r97test::DeviceGuard g(o); });
    R97_CHECK(why.find("card1") != std::string::npos);
    R97_CHECK(why.find("70 MiB") != std::string::npos);      // the free amount, named
    R97_CHECK(why.find("1024 MiB") != std::string::npos);    // and the needed one
    R97_CHECK(why.find("pid 999") != std::string::npos);     // and who holds the card
    R97_CHECK(why.find("radiance") != std::string::npos);

    // The refused constructor released its lock, and the same card with enough VRAM passes.
    std::ofstream(fs::path(drm.path()) / "card1/device/mem_info_vram_used") << "0\n";
    o.fail_fast = true;  // would refuse if the failed guard still held the lock
    r97test::DeviceGuard fits(o);
    R97_CHECK(fits.card() == "card1");
}

R97_TEST(holder_parsing) {
    // What `amd-smi process --csv` writes: the header, then one line per process with the record
    // nested in one quoted Python-dict field; a card with no holders reads as the last row.
    const std::string csv =
        "gpu,process_info\n"
        "0,\"{'name': 'radiance', 'pid': 1543186, 'memory_usage': {'gtt_mem': 0, 'cpu_mem': 0, "
        "'vram_mem': 34087321600}, 'mem_usage': 0, 'usage': {'gfx': 0, 'enc': 0}, 'sdma_usage': "
        "0, 'cu_occupancy': 0, 'evicted_time': 0}\"\n"
        "1,\"{'name': 'other', 'pid': 42, 'memory_usage': {'gtt_mem': 0, 'cpu_mem': 0, "
        "'vram_mem': 8000000000}, 'mem_usage': 0, 'usage': {'gfx': 0, 'enc': 0}, 'sdma_usage': "
        "0, 'cu_occupancy': 0, 'evicted_time': 0}\"\n"
        "1,\"{'name': 'idle', 'pid': 7, 'memory_usage': {'gtt_mem': 0, 'cpu_mem': 0, "
        "'vram_mem': 0}, 'mem_usage': 0, 'usage': {'gfx': 0, 'enc': 0}, 'sdma_usage': 0, "
        "'cu_occupancy': 0, 'evicted_time': 0}\"\n"
        "0,No running processes detected\n"
        "not,a,parser\n";
    const std::vector<Holder> holders = r97test::parse_holders(csv);
    R97_CHECK(holders.size() == 2);
    R97_CHECK(holders[0].pid == 1543186 && holders[0].vram == 34087321600LL);
    R97_CHECK(holders[0].name == "radiance" && holders[0].gpu == 0);
    R97_CHECK(holders[1].pid == 42 && holders[1].name == "other" && holders[1].gpu == 1);

    // The refusal text names every holder with its pid and VRAM.
    const std::string text =
        r97test::vram_short_text("card1", 70 * r97test::kMiB, 1024 * r97test::kMiB, holders);
    R97_CHECK(text.find("70 MiB") != std::string::npos);
    R97_CHECK(text.find("1024 MiB") != std::string::npos);
    R97_CHECK(text.find("pid 1543186 radiance") != std::string::npos);

    // No holders listed: the message says where to look instead.
    const std::string none =
        r97test::vram_short_text("card1", 70 * r97test::kMiB, 1024 * r97test::kMiB, {});
    R97_CHECK(none.find("amd-smi") != std::string::npos);

    // A name with an apostrophe makes Python repr the string with double quotes, which the CSV
    // writer escapes as ""; both the field and the record still parse.
    const std::string quoted =
        "gpu,process_info\n"
        "0,\"{'name': \"\"don't panic\"\", 'pid': 5, 'memory_usage': {'vram_mem': 1234}}\"\n";
    const std::vector<Holder> one = r97test::parse_holders(quoted);
    R97_CHECK(one.size() == 1);
    R97_CHECK(one[0].pid == 5 && one[0].vram == 1234);
    R97_CHECK(one[0].name == "don't panic");
}

// Waits for the child, giving up after a few seconds rather than wedging the tier when something
// the guard does deadlocks.
int wait_child(pid_t pid, const char* what) {
    for (int i = 0; i < 500; ++i) {
        int st = 0;
        const pid_t got = waitpid(pid, &st, WNOHANG);
        if (got == pid)
            return st;
        if (got < 0)
            throw std::runtime_error(std::string("waitpid ") + what + ": " + std::strerror(errno));
        usleep(10 * 1000);
    }
    kill(pid, SIGKILL);
    (void)waitpid(pid, nullptr, 0);
    throw std::runtime_error(std::string(what) + ": did not finish in 5 s");
}

R97_TEST(lock_mutual_exclusion) {
    const TempDir drm;
    fake_drm_one_card(drm);  // the lock test is about locking, not card selection
    const TempDir locks;

    GuardOptions o;
    o.device = 123;  // not a real card's number
    o.need_vram = 0;
    o.sysfs_root = drm.path();
    o.lock_dir = locks.path();

    int fds[2];
    pid_t pid = -1;
    {
        r97test::DeviceGuard first(o);
        R97_CHECK(first.lock_path() == locks.path() + "/r97-card123.lock");

        // Another run on the same card is refused at once when it cannot queue.
        GuardOptions b = o;
        b.fail_fast = true;
        const std::string why = thrown([&] { r97test::DeviceGuard second(b); });
        R97_CHECK(why.find("held by another run") != std::string::npos);

        // And it queues when it can: the child blocks until the parent lets go.
        R97_CHECK(pipe(fds) == 0);
        pid = fork();
        R97_CHECK(pid >= 0);
        if (pid == 0) {
            close(fds[0]);
            // Drop everything inherited but the pipe: a forked child that kept the parent's lock
            // descriptor would queue against itself (the lock fd is O_CLOEXEC, so exec drops it,
            // but a fork without exec does not).
            for (int drop = 3; drop < 1024; ++drop)
                if (drop != fds[1])
                    close(drop);
            try {
                r97test::DeviceGuard queued(o);
                (void)!write(fds[1], "ok", 2);
                _exit(0);
            } catch (...) {
                _exit(33);
            }
        }
        close(fds[1]);

        // Still held: no byte yet within 200 ms.
        pollfd pf{fds[0], POLLIN, 0};
        R97_CHECK(poll(&pf, 1, 200) == 0);
    }  // first's destructor releases the card; the queued child gets it, writes and exits

    const int st = wait_child(pid, "lock_mutual_exclusion's queued child");
    R97_CHECK(WIFEXITED(st));
    R97_CHECK(WEXITSTATUS(st) == 0);
    char got[3] = {0};
    R97_CHECK(read(fds[0], got, 2) == 2);
    R97_CHECK(std::string(got) == "ok");
    close(fds[0]);

    // The lock is free again for a fresh guard.
    r97test::DeviceGuard again(o);
    R97_CHECK(again.lock_path() == locks.path() + "/r97-card123.lock");
}

R97_TEST(lock_across_users) {
    // Several people on one machine: whoever's umask created the lock file, another user's run
    // must queue on it rather than fail to open it. Tested by becoming the other user, which needs
    // root; a non-root run cannot test this and says so.
    if (geteuid() != 0) {
        std::fprintf(stderr, "  (not root: the two-user lock check is not run)\n");
        return;
    }
    const TempDir drm;
    fake_drm_one_card(drm);
    const TempDir locks;
    R97_CHECK(chmod(drm.path().c_str(), 0755) == 0);    // traversable by the other user
    R97_CHECK(chmod(locks.path().c_str(), 0755) == 0);

    GuardOptions o;
    o.device = 321;  // not a real card's number
    o.need_vram = 0;
    o.sysfs_root = drm.path();
    o.lock_dir = locks.path();
    const std::string lock = locks.path() + "/r97-card321.lock";

    int fds[2];
    pid_t pid = -1;
    {
        r97test::DeviceGuard first(o);
        // As restrictive as an odd umask can leave it: readable, not writable, for anyone else.
        // The guard's read-only fallback is what makes the lock work anyway.
        R97_CHECK(chmod(lock.c_str(), 0444) == 0);

        R97_CHECK(pipe(fds) == 0);
        pid = fork();
        R97_CHECK(pid >= 0);
        if (pid == 0) {
            close(fds[0]);
            for (int drop = 3; drop < 1024; ++drop)
                if (drop != fds[1])
                    close(drop);
            if (setgid(65534) != 0 || setuid(65534) != 0)
                _exit(44);  // cannot become the other user here
            try {
                r97test::DeviceGuard queued(o);  // on a file it cannot write
                (void)!write(fds[1], "ok", 2);
                _exit(0);
            } catch (...) {
                _exit(33);  // e.g. "cannot open the card lock ... Permission denied"
            }
        }
        close(fds[1]);

        // Still held: no byte yet.
        pollfd pf{fds[0], POLLIN, 0};
        R97_CHECK(poll(&pf, 1, 200) == 0);
    }

    const int st = wait_child(pid, "lock_across_users' child");
    R97_CHECK(WIFEXITED(st));
    if (WEXITSTATUS(st) == 44) {
        std::fprintf(stderr, "  (cannot become another user here: the two-user lock check is not run)\n");
    } else {
        R97_CHECK(WEXITSTATUS(st) == 0);
        char got[3] = {0};
        R97_CHECK(read(fds[0], got, 2) == 2);
        R97_CHECK(std::string(got) == "ok");
    }
    close(fds[0]);
}

R97_TEST(model_container) {
    const TempDir dir;  // empty: no container
    R97_CHECK(setenv("R97_MODEL_DIR", dir.path().c_str(), 1) == 0);

    // An empty directory: a skip with the message that says what to do, nothing more.
    std::string why;
    R97_CHECK(!r97test::resolve_container("", &why));
    R97_CHECK(why.find(dir.path()) != std::string::npos);
    R97_CHECK(why.find("hf download") != std::string::npos);

    const fs::path rad = fs::path(dir.path()) / "qwen3.8-27b-iq4r-s-v4.rad";
    std::ofstream(rad) << "container";
    R97_CHECK(r97test::resolve_container() == rad.string());

    // The variable names the .rad file directly, too.
    R97_CHECK(setenv("R97_MODEL_DIR", rad.c_str(), 1) == 0);
    R97_CHECK(r97test::resolve_container() == rad.string());

    // Unset: the build's R97_MODEL_RAD macro or the default directory, whichever is there.
    R97_CHECK(unsetenv("R97_MODEL_DIR") == 0);
    const std::string explicit_rad = r97test::resolve_container("", &why).value_or("");
    R97_CHECK(explicit_rad == R97_MODEL_RAD || why.find("hf download") != std::string::npos);
}

R97_TEST(model_skip_exit) {
    // A model test with no container prints why and ends the tier with kSkip (77), which ctest
    // records as skipped (SKIP_RETURN_CODE in CMakeLists.txt), not as a failure. In a child, so
    // that the rest of the host tier still runs.
    const TempDir dir;
    int fds[2];
    R97_CHECK(pipe(fds) == 0);
    const pid_t pid = fork();
    R97_CHECK(pid >= 0);
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        setenv("R97_MODEL_DIR", dir.path().c_str(), 1);
        r97test::container_or_skip();  // never returns
        _exit(99);
    }
    close(fds[1]);
    std::string msg;
    char buf[512];
    ssize_t n;
    while ((n = read(fds[0], buf, sizeof buf)) > 0)
        msg.append(buf, (size_t)n);
    close(fds[0]);
    const int st = wait_child(pid, "model_skip_exit's child");
    R97_CHECK(WIFEXITED(st));
    R97_CHECK(WEXITSTATUS(st) == r97test::kSkip);
    R97_CHECK(msg.find("skip: ") == 0);
    R97_CHECK(msg.find(dir.path()) != std::string::npos);
    R97_CHECK(msg.find("hf download") != std::string::npos);
}
