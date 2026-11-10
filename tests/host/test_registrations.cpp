// tests/host/test_registrations.cpp -- the host binary's static registrations (label: host).
//
// The host tier is one binary over every tests/host/*.cpp, so a registration belongs in exactly
// one translation unit: registering the same name twice aborts at static-init time. Everything
// the tests look up by name is registered here: the test objective and test delta, and the
// subcommands that print them. The tests themselves register with R97_TEST in their own files.
#include "../support/test.h"

#include "r97/model.h"

#include "../support/bias_delta.h"
#include "../support/sum_objective.h"

R97_REGISTER_OBJECTIVE(test_sum, r97test::make_sum_objective);
R97_REGISTER_DELTA(test_bias, r97test::make_bias_delta);

// The host tier links r97 but not tools/. These are the command translation units that need no
// model; r97train's main stays out. They register with the same table r97train dispatches.
#include "../../tools/commands.cpp"
#include "../../tools/deltas.cpp"
#include "../../tools/objectives.cpp"
