// tests/host/test_runner.cpp -- the one main the host-tests binary is allowed to have; every
// other tests/host/*.cpp registers itself through tests/support/test_reg.h.
#include "support/test_reg.h"

int main() { return r97test::run_all(); }
