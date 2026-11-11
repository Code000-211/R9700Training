// src/adjoint_family.h -- internal: the adjoint family an op's backward belongs to.
//
// The family names the doc file (docs/adjoints/<family>.md) and the source file
// (src/adjoint/host/<family>.cpp) an op's backward lives in, so the inventory can tell a reader
// working an adjoint issue which ops are already theirs. An op that fits no family is reported as
// None rather than guessed into a family whose backward it does not share: a name that looks like
// a family's is not evidence that the backward is there.
//
// The claims themselves are one table in src/adjoint_family.cpp, because the ops it has to claim
// belong to the adjoint family files other people are working in, and a parallel change adds files
// instead of editing them (CONTRIBUTING.md). A new op whose backward is in its own file may claim
// its family there instead, with the macro below, and nothing here has to change.
#pragma once

#include <initializer_list>
#include <string>

namespace r97 {

enum class AdjointFamily {
    None,
    Elementwise,
    Linear,
    Quant,
    Lora,
    Qk,
    Attention,
    GdnPrep,
    GdnKkt,
    GdnScan,
};

const char* adjoint_family_name(AdjointFamily family);

// The family `op`'s backward belongs to, or None when no file claimed it.
AdjointFamily family_find(const std::string& op);

// Claim `ops` for `family`. Called from the static registrars the macro below makes; an op claimed
// twice for different families is a hard error, not a silent winner.
void family_register(AdjointFamily family, std::initializer_list<const char*> ops);

struct FamilyRegistrar {
    FamilyRegistrar(AdjointFamily family, std::initializer_list<const char*> ops) {
        family_register(family, ops);
    }
};

// The registrar's name comes from the line, so one file may claim several groups of ops.
#define R97_FAMILY_NAME2(a, b) a##b
#define R97_FAMILY_NAME(a, b) R97_FAMILY_NAME2(a, b)
#define R97_REGISTER_ADJOINT_FAMILY(family, ...) \
    static ::r97::FamilyRegistrar R97_FAMILY_NAME(r97_reg_family_, __LINE__) { \
        ::r97::AdjointFamily::family, {__VA_ARGS__} \
    }

}  // namespace r97
