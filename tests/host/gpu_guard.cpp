// tests/host/gpu_guard.cpp -- the GPU test guard's logic against a fake filesystem, no card.
#include "support/gpu_guard.h"
#include "support/model_dir.h"
#include "support/r97_test.h"

#undef NDEBUG
#include <cassert>

#include <poll.h>
#include <sys/wait.h>

#include <cerrno>
#include <cstdio>
#include <fstream>
#include <string>

using r97test::CardVram;
using r97test::GuardOptions;
using r97test::Holder;

// A fake /sys/class/drm: card0 (no VRAM of its own), card1, card2, card10, in a temp dir.
static std::string fake_sysfs() {
    char dir[] = "/tmp/r97-guard-fake-XXXXXX";
    assert(mkdtemp(dir));
    std::string root = dir;
    auto write = [root](const std::string& path, const std::string& text) {
        std::string full = root + "/" + path;
        std::string cmd = "mkdir -p " + full.substr(0, full.rfind('/'));
        assert(system(cmd.c_str()) == 0);
        std::ofstream(full) << text << "\n";
    };
    write("card0/device/uevent", "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:03:00.0\n");
    write("card1/device/mem_info_vram_total", "34225520640");
    write("card1/device/mem_info_vram_used", "34151870464");
    write("card1/device/uevent", "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:43:00.0\n");
    write("card2/device/mem_info_vram_total", "34225520640");
    write("card2/device/mem_info_vram_used", "0");
    write("card2/device/uevent", "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:c3:00.0\n");
    write("card10/device/mem_info_vram_total", "34225520640");
    write("card10/device/mem_info_vram_used", "0");
    write("card10/device/uevent", "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:d3:00.0\n");
    return root;
}

R97_TEST(parse_device) {
    assert(r97test::parse_device("0") == 0);
    assert(r97test::parse_device("3") == 3);
    assert(r97test::parse_device(" 7 ") == 7);
    for (const char* bad : {"", "-1", "x", "1x", "one"})
        try {
            r97test::parse_device(bad);
            assert(false);
        } catch (const std::exception& e) {
            assert(std::string(e.what()).find("R97_DEVICE") != std::string::npos);
        }
}

R97_TEST(sysfs_vram) {
    std::string root = fake_sysfs();
    CardVram v = r97test::read_card_vram(root, "card1");
    assert(v.total == 34225520640LL && v.used == 34151870464LL);
    // Only cards with their own VRAM list: the connectors (card0) and a missing root do not.
    std::vector<std::string> cards = r97test::list_cards(root);
    assert((cards == std::vector<std::string>{"card1", "card2", "card10"}));
    try {
        r97test::read_card_vram(root, "card0");
        assert(false);
    } catch (const std::exception& e) {
        assert(std::string(e.what()).find("mem_info_vram_total") != std::string::npos);
    }
    assert(system(("rm -rf " + root).c_str()) == 0);
}

R97_TEST(bdf_resolution) {
    std::string root = fake_sysfs();
    assert(r97test::card_pci_slot(root, "card1") == "0000:43:00.0");
    assert(r97test::card_for_bdf(root, "0000:43:00.0") == "card1");
    assert(r97test::card_for_bdf(root, "0000:C3:00.0") == "card2");
    assert(r97test::card_for_bdf(root, "0000:99:00.0").empty());
    // Two cards and no bdf: the guard names the problem instead of checking the wrong card.
    GuardOptions o;
    o.sysfs_root = root;
    o.need_vram = 0;
    try {
        r97test::DeviceGuard two(o);
        assert(false);
    } catch (const std::exception& e) {
        assert(std::string(e.what()).find("GuardOptions::bdf") != std::string::npos);
    }
    // But a single-card machine is unambiguous without a bdf.
    std::string one = root + "-one";
    assert(system(("mkdir -p " + one + "/card2/device").c_str()) == 0);
    std::ofstream(one + "/card2/device/mem_info_vram_total") << "1\n";
    std::ofstream(one + "/card2/device/mem_info_vram_used") << "0\n";
    o.sysfs_root = one;
    r97test::DeviceGuard ok(o);
    assert(ok.card() == "card2");
    assert(system(("rm -rf " + root + " " + one).c_str()) == 0);
}

R97_TEST(lock_mutual_exclusion) {
    std::string locks = "/tmp/r97-guard-locks-test";
    assert(system(("rm -rf " + locks).c_str()) == 0);
    // An empty sysfs root: the lock test is about locking, not card resolution.
    std::string no_cards = "/tmp/r97-guard-locks-sysfs";
    assert(system(("rm -rf " + no_cards + " && mkdir -p " + no_cards).c_str()) == 0);
    GuardOptions o;
    o.device = 123;  // not a real card's number
    o.lock_dir = locks;
    o.sysfs_root = no_cards;
    // Shared with the block below: the forked child and its report pipe.
    int fds[2];
    pid_t pid = -1;
    {
        r97test::DeviceGuard first(o);
        assert(first.lock_path() == locks + "/r97-card123.lock");
        // Another run on the same card is refused at once when it cannot queue.
        GuardOptions b = o;
        b.fail_fast = true;
        try {
            r97test::DeviceGuard second(b);
            assert(false);
        } catch (const std::exception& e) {
            assert(std::string(e.what()).find("held by another run") != std::string::npos);
        }
        // And it queues when it can: the child blocks until the parent lets go.
        assert(pipe(fds) == 0);
        pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            close(fds[0]);
            // Drop everything inherited but the pipe: a forked child that keeps the parent's
            // lock descriptor would queue against itself (the lock fd is O_CLOEXEC, so exec
            // drops it, but a fork without exec does not).
            for (int drop = 3; drop < 1024; ++drop)
                if (drop != fds[1])
                    close(drop);
            try {
                r97test::DeviceGuard queued(o);
                (void)!write(fds[1], "ok", 2);
                close(fds[1]);
                _exit(0);
            } catch (...) {
                _exit(33);
            }
        }
        close(fds[1]);
        // Still held: no byte yet within 100 ms.
        pollfd pf{fds[0], POLLIN, 0};
        assert(poll(&pf, 1, 100) == 0);
    }  // first's destructor releases the card; the queued child gets it, writes and exits
    {
        // The queued child now ran past its blocking constructor and left the lock free again.
        int st = 0;
        pid_t reaped = waitpid(pid, &st, 0);
        if (reaped != pid)
            fprintf(stderr, "lock_mutual_exclusion: waitpid(%d) -> %d, errno=%d\n", (int)pid,
                   (int)reaped, errno); fflush(stderr);
        assert(reaped == pid);
        assert(WIFEXITED(st) && WEXITSTATUS(st) == 0);
        char got[3] = {0};
        assert(read(fds[0], got, 2) == 2 && std::string(got) == "ok");
        close(fds[0]);
        // And the lock is free for a fresh guard.
        r97test::DeviceGuard again(o);
        assert(again.lock_path() == locks + "/r97-card123.lock");
    }
    r97test::DeviceGuard last(o);
    assert(system(("rm -rf " + locks + " " + no_cards).c_str()) == 0);
}

R97_TEST(vram_refusal) {
    std::string root = fake_sysfs();  // card1: 32644 MiB total, 32575 MiB used -> 70 MiB free
    // A stub amd-smi, printing what `amd-smi process -g 0 --csv` prints on a held card: one line
    // per process, the record nested in one quoted field.
    std::string bin = "/tmp/r97-guard-stub-amd-smi";
    std::ofstream(bin) << "#!/bin/sh\n"
                          "cat <<'R97STUB'\n"
                          "gpu,process_info\n"
                          "0,\"{'name': 'radiance', 'pid': 999, 'memory_usage': {'gtt_mem': 0, "
                          "'cpu_mem': 0, 'vram_mem': 33280000000}, 'mem_usage': 0, "
                          "'usage': {'gfx': 0, 'enc': 0}, 'sdma_usage': 0, 'cu_occupancy': 0, "
                          "'evicted_time': 0}\"\n"
                          "R97STUB\n";
    assert(system(("chmod +x " + bin).c_str()) == 0);

    GuardOptions o;
    o.sysfs_root = root;
    o.device = 0;
    o.bdf = "0000:43:00.0";
    o.need_vram = 1 << 30;  // 1 GiB, against 70 MiB free
    o.amd_smi = bin;
    try {
        r97test::DeviceGuard guard(o);
        assert(false);
    } catch (const std::exception& e) {
        std::string text = e.what();
        assert(text.find("card1") != std::string::npos);
        assert(text.find("70 MiB") != std::string::npos);   // the free amount, named
        assert(text.find("1024 MiB") != std::string::npos); // and the needed one
        assert(text.find("pid 999") != std::string::npos);  // and who holds the card
    }

    // The same card with enough VRAM passes the check.
    std::ofstream(root + "/card1/device/mem_info_vram_used") << "0\n";
    r97test::DeviceGuard fits(o);
    assert(fits.card() == "card1");
    std::remove(bin.c_str());
    assert(system(("rm -rf " + root).c_str()) == 0);
}

R97_TEST(holder_parsing) {
    // What `amd-smi process --csv` writes: the header, then one line per process with the record
    // nested in one quoted Python-dict field; a card with no holders reads as the last row.
    std::string csv =
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
    std::vector<Holder> holders = r97test::parse_holders(csv);
    assert(holders.size() == 2);
    assert(holders[0].pid == 1543186 && holders[0].vram == 34087321600LL);
    assert(holders[0].name == "radiance" && holders[0].gpu == 0);
    assert(holders[1].pid == 42 && holders[1].name == "other" && holders[1].gpu == 1);
    // The refusal text names an unknown holder's process when /proc has it, "?" when not.
    std::string text = r97test::vram_short_text("card1", 70 * r97test::kMiB,
                                                1024 * r97test::kMiB, holders);
    assert(text.find("70 MiB") != std::string::npos && text.find("1024 MiB") != std::string::npos);
    assert(text.find("1543186") != std::string::npos);
    // No holders listed: the message says where to look instead.
    std::string none = r97test::vram_short_text("card1", 70 * r97test::kMiB,
                                                1024 * r97test::kMiB, {});
    assert(none.find("amd-smi") != std::string::npos);

    // A name with an apostrophe makes Python repr the string with double quotes, which the CSV
    // writer escapes as ""; both the field and the record still parse.
    std::string quoted =
        "gpu,process_info\n"
        "0,\"{'name': \"\"don't panic\"\", 'pid': 5, 'memory_usage': {'vram_mem': 1234}}\"\n";
    holders = r97test::parse_holders(quoted);
    assert(holders.size() == 1);
    assert(holders[0].pid == 5 && holders[0].vram == 1234);
    assert(holders[0].name == "don't panic");
}

R97_TEST(model_container) {
    std::string dir = "/tmp/r97-model-dir-test";
    assert(system(("rm -rf " + dir).c_str()) == 0);
    assert(mkdir(dir.c_str(), 0777) == 0);
    setenv("R97_MODEL_DIR", dir.c_str(), 1);

    // An empty directory: a skip with the message that says what to do, nothing more.
    std::string why;
    assert(!r97test::resolve_container("", &why));
    assert(why.find(dir) != std::string::npos);
    assert(why.find("hf download") != std::string::npos);

    std::ofstream(dir + "/qwen3.8-27b-iq4r-s-v4.rad") << "container";
    assert(r97test::resolve_container() == dir + "/qwen3.8-27b-iq4r-s-v4.rad");

    // The variable names the .rad file directly, too.
    setenv("R97_MODEL_DIR", (dir + "/qwen3.8-27b-iq4r-s-v4.rad").c_str(), 1);
    assert(r97test::resolve_container() == dir + "/qwen3.8-27b-iq4r-s-v4.rad");

    // Unset: the build's R97_MODEL_RAD macro or the default directory, whichever is there.
    unsetenv("R97_MODEL_DIR");
    std::string explicit_rad = r97test::resolve_container("", &why).value_or("");
    assert(explicit_rad == R97_MODEL_RAD || why.find("hf download") != std::string::npos);
    assert(system(("rm -rf " + dir).c_str()) == 0);
}
