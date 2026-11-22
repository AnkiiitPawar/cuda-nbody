#pragma once

#include <cstdio>

namespace testing {
inline int failures = 0;
inline void check(bool ok, const char* name) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}
}  // namespace testing
