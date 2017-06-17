//
// Created by salmon on 17-6-17.
//

#ifndef SIMPLA_C_14_PORT_H
#define SIMPLA_C_14_PORT_H

#include <type_traits>

#if __cplusplus < 201402L
namespace std {
template <bool BOOLEAN, typename... T>
using conditional_t = typename conditional<BOOLEAN, T...>::type;
template <bool BOOLEAN, typename... T>
using enable_if_t = typename enable_if<BOOLEAN, T...>::type;
}

#define AUTO_RETURN(_EXPR_) \
    ->decltype((_EXPR_)) { return (_EXPR_); }

#else

#define AUTO_RETURN(_EXPR_) \
    { return (_EXPR_); }

#endif

#endif  // SIMPLA_C_14_PORT_H
