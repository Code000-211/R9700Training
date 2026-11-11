// The adjoint contract header compiles and its types stand on their own.
// Registration is exercised by the host tests (tests/host/adjoint_add.cpp) and
// by src/adjoint/host/add.cpp: the macros need the registry's object file at
// link time, which a header-only check does not carry.
#include "r97/adjoint.h"

namespace {

void hdr_twin(const r97::OpRecord&, const std::vector<r97::AdValue>&, std::vector<r97::AdValue>*) {}
bool hdr_pred(const std::vector<r97::OpParam>&) { return true; }
void hdr_adj(const r97::AdjointCall&) {}

r97::TwinFn tw = hdr_twin;
r97::AdjointFn ad = hdr_adj;
r97::ParamPred pr = hdr_pred;

}  // namespace

int main() { return tw && ad && pr ? 0 : 1; }
