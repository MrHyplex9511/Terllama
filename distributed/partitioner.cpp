/*
 * partitioner.cpp — translation unit for partitioner.h.
 *
 * All logic is header-only (inline); this TU exists so the header is
 * syntax-checked as a standalone unit and to give the build system a
 * translation unit to compile.
 */
#include "partitioner.h"

// Force instantiation/odr-use of the inline functions so a lone
// -fsyntax-only compile still fully parses them.
namespace tldist {
namespace {
volatile int keep_alive = 0;
}
}  // namespace tldist
