//
// Created by salmon on 17-6-24.
//

#ifndef SIMPLA_ARRAYNTUPLE_H
#define SIMPLA_ARRAYNTUPLE_H

#include "Array.h"
#include "nTuple.h"
namespace simpla {

#define _SP_DEFINE_ARRAY_BINARY_OPERATOR(_OP_, _NAME_)                                      \
    template <typename... TL, typename TR, int... NR>                                       \
    auto operator _OP_(Array<TL...> const& lhs, nTuple<TR, NR...> const& rhs) {             \
        return Expression<simpla::tags::_NAME_, Array<TL...>, nTuple<TR, NR...>>(lhs, rhs); \
    };                                                                                      \
    template <typename TL, int... NL, typename... TR>                                       \
    auto operator _OP_(nTuple<TL, NL...> const& lhs, Array<TR...> const& rhs) {             \
        return Expression<simpla::tags::_NAME_, nTuple<TL, NL...>, Array<TR...>>(lhs, rhs); \
    };

_SP_DEFINE_ARRAY_BINARY_OPERATOR(+, addition)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(-, subtraction)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(*, multiplication)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(/, division)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(%, modulo)

#undef _SP_DEFINE_ARRAY_BINARY_OPERATOR

// template <typename... T, int... N, typename TFun>
// auto foreach (nTuple<Array<T...>, N...> const& v, TFun const& f) {
//    foreach (v, [&](auto& a, auto&&... subs) {
//        foreach (a, [&](auto& u, auto&&... idx) {
//            u = f(std::forward<decltype(subs)>(subs)..., std::forward<decltype(idx)>(idx)...);
//        })
//            ;
//    })
//        ;
//}

}  // namespace simpla {

#endif  // SIMPLA_ARRAYNTUPLE_H
