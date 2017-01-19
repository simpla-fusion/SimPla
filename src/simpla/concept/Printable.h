//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_PRINTABLE_H
#define SIMPLA_PRINTABLE_H

#include <ostream>
#include "CheckConcept.h"

namespace simpla {
namespace concept { /**  @ingroup concept   */

/**
 * @brief a type whose instances maybe print to std::ostream  *
 * @details
 * ## Summary
 *
 * ## Requirements
 *  Class \c R implementing the concept of @ref Printable must define:
 *   Pseudo-Signature                                      | Semantics
 *	 ------------------------------------------------------|----------
 * 	 \code  std::ostream &Print(std::ostream &os, int indent)   \endcode |
 */
struct Printable {
    virtual std::ostream &Print(std::ostream &os, int indent) const { return os; };
};

CHECK_FUNCTION_MEMBER(is_printable, print);

inline std::ostream &operator<<(std::ostream &os, Printable const &obj) {
    obj.Print(os, 0);
    return os;
}
}
}
#endif  // SIMPLA_PRINTABLE_H
