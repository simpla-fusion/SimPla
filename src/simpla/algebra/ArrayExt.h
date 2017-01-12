//
// Created by salmon on 16-12-28.
//

#ifndef SIMPLA_ARRAYEXT_H
#define SIMPLA_ARRAYEXT_H

#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>
#include <simpla/mpl/any.h>
#include <simpla/concept/CheckConcept.h>
#include <simpla/toolbox/FancyStream.h>

#include "Array.h"

namespace simpla { namespace algebra
{
namespace declare
{
template<typename T, size_type NDIMS, bool IS_SLOW_FIRST>
std::ostream &operator<<(std::ostream &os, Array_<T, NDIMS, IS_SLOW_FIRST> const &v)
{
    printNdArray(os, v.m_data_, NDIMS, v.m_dims_);
    return os;
}
//template<typename T, size_type  ...M>
//std::istream &operator>>(std::istream &os, nTuple_<T, M...> &v) { return input(os, v); }
}

}}//namespace simpla{namespace algebra
#endif //SIMPLA_ARRAYEXT_H
