//
// Created by salmon on 17-6-24.
//

#ifndef SIMPLA_ARRAYNTUPLE_H
#define SIMPLA_ARRAYNTUPLE_H

#include "Array.h"
#include "nTuple.h"
namespace simpla {

#define _SP_DEFINE_ARRAY_BINARY_OPERATOR(_OP_, _NAME_)                                            \
    template <typename TL, int NL, typename SFC, typename TR, int... NR>                          \
    auto operator _OP_(Array<TL, NL, SFC> const& lhs, nTuple<TR, NR...> const& rhs) {             \
        return Expression<simpla::tags::_NAME_, Array<TL, NL, SFC>, nTuple<TR, NR...>>(lhs, rhs); \
    };                                                                                            \
    template <typename TL, int... NL, typename TR, int NR, typename SFC>                          \
    auto operator _OP_(nTuple<TL, NL...> const& lhs, Array<TR, NR, SFC> const& rhs) {             \
        return Expression<simpla::tags::_NAME_, nTuple<TL, NL...>, Array<TR, NR, SFC>>(lhs, rhs); \
    };

_SP_DEFINE_ARRAY_BINARY_OPERATOR(+, addition)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(-, subtraction)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(*, multiplication)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(/, division)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(%, modulo)

#undef _SP_DEFINE_ARRAY_BINARY_OPERATOR
}
#endif  // SIMPLA_ARRAYNTUPLE_H
