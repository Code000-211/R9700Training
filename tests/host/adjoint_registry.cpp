// tests/host/adjoint_registry.cpp -- the lookup rules behind r97/adjoint.h, which every adjoint
// issue inherits: a row's predicate picks it by the parameters, the side picks the host row from
// the device one, the frozen positions travel with the row that serves the op rather than with the
// op name, and an op that nothing serves says so by name and by parameters. Label: host.
#include "support/test.h"

#include "r97/adjoint.h"

#include <algorithm>
#include <string>
#include <variant>
#include <vector>

namespace {

using r97::AdValue;
using r97::AdjointCall;
using r97::OpParam;
using r97::OpRecord;
using r97::Side;

// One name no op and no other test uses, so the rows other files register cannot answer here.
constexpr const char* kProbe = "registry_probe";

void probe_twin_bf16(const OpRecord&, const std::vector<AdValue>&, std::vector<AdValue>*) {}
void probe_twin_any(const OpRecord&, const std::vector<AdValue>&, std::vector<AdValue>*) {}
void probe_adj_bf16(const AdjointCall&) {}
void probe_adj_any(const AdjointCall&) {}
void probe_adj_device(const AdjointCall&) {}

bool is_bf16(const std::vector<OpParam>& params) {
    for (const OpParam& p : params)
        if (p.first == "dtype") {
            const std::string* dtype = std::get_if<std::string>(&p.second);
            return dtype != nullptr && *dtype == "bf16";
        }
    return false;
}

// The catch-all rows come last: rows are tried in registration order, and the first one whose
// predicate accepts the parameters serves the record.
R97_REGISTER_TWIN_IF(kProbe, probe_twin_bf16, is_bf16)
R97_REGISTER_TWIN(kProbe, probe_twin_any)
R97_REGISTER_ADJOINT_IF_F(kProbe, host, probe_adj_bf16, is_bf16, 2)
R97_REGISTER_ADJOINT(kProbe, host, probe_adj_any)
R97_REGISTER_ADJOINT_F(kProbe, device, probe_adj_device, 0, 1)

std::vector<OpParam> params(const char* dtype) {
    return {{"M", int64_t{4}}, {"n", int64_t{8}}, {"dtype", std::string(dtype)}};
}

}  // namespace

R97_TEST(adjoint_registry_picks_the_row_by_predicate) {
    std::string error;
    R97_CHECK(r97::twin_find(kProbe, params("bf16"), &error) == probe_twin_bf16);
    R97_CHECK(error.empty());
    R97_CHECK(r97::twin_find(kProbe, params("w8a8"), &error) == probe_twin_any);
    R97_CHECK(error.empty());
    R97_CHECK(r97::adjoint_find(kProbe, Side::host, params("bf16"), &error) == probe_adj_bf16);
    R97_CHECK(error.empty());
    R97_CHECK(r97::adjoint_find(kProbe, Side::host, params("w8a8"), &error) == probe_adj_any);
    R97_CHECK(error.empty());
}

R97_TEST(adjoint_registry_picks_the_side) {
    std::string error;
    R97_CHECK(r97::adjoint_find(kProbe, Side::device, params("bf16"), &error) ==
              probe_adj_device);
    R97_CHECK(error.empty());
    R97_CHECK(r97::adjoint_find(kProbe, Side::host, params("bf16"), &error) == probe_adj_bf16);
}

R97_TEST(adjoint_registry_frozen_positions_come_from_the_serving_row) {
    R97_CHECK(r97::adjoint_frozen(kProbe, Side::host, params("bf16")) == std::vector<int>{2});
    R97_CHECK(r97::adjoint_frozen(kProbe, Side::host, params("w8a8")).empty());
    R97_CHECK(r97::adjoint_frozen(kProbe, Side::device, params("bf16")) ==
              (std::vector<int>{0, 1}));
}

R97_TEST(adjoint_registry_lists_what_it_serves) {
    const std::vector<std::string> names = r97::ops_registered();
    R97_CHECK(std::find(names.begin(), names.end(), kProbe) != names.end());
    R97_CHECK(std::is_sorted(names.begin(), names.end()));

    const r97::OpRows rows = r97::op_rows(kProbe);
    R97_CHECK(rows.twin);
    R97_CHECK(rows.host_adjoint);
    R97_CHECK(rows.device_adjoint);
    R97_CHECK(rows.host_frozen == (std::vector<int>{2}));
    R97_CHECK(rows.device_frozen == (std::vector<int>{0, 1}));
}

R97_TEST(adjoint_registry_names_the_op_and_its_parameters_when_nothing_serves_it) {
    const std::vector<OpParam> p = {{"M", int64_t{3}}, {"dtype", std::string("bf16")}};
    std::string twin_error;
    std::string adjoint_error;

    R97_CHECK(r97::twin_find("registry_probe_none", p, &twin_error) == nullptr);
    R97_CHECK(twin_error.find("registry_probe_none") != std::string::npos);
    R97_CHECK(twin_error.find("M=3") != std::string::npos);
    R97_CHECK(twin_error.find("dtype=bf16") != std::string::npos);

    R97_CHECK(r97::adjoint_find("registry_probe_none", Side::device, p, &adjoint_error) == nullptr);
    R97_CHECK(adjoint_error.find("device adjoint") != std::string::npos);
    R97_CHECK(adjoint_error.find("registry_probe_none") != std::string::npos);
    R97_CHECK(adjoint_error.find("M=3") != std::string::npos);
}
