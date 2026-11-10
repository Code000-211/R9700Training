// tests/host/test_main.cpp -- the single main the r97_host_tests binary needs; it runs
// every registered test (see tests/support/test.h).
#include "support/test.h"

int main() { return r97test::run_all(); }
