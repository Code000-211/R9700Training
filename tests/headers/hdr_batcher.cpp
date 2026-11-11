// tests/headers/hdr_batcher.cpp -- r97/batcher.h alone compiles and type-checks.
#include "r97/batcher.h"

int main() {
    r97::BatcherConfig cfg{};
    r97::Batcher::Position pos{};
    (void)cfg;
    (void)pos;
    return 0;
}
