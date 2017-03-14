//
// Created by salmon on 17-3-3.
//

#ifndef SIMPLA_ANY_H
#define SIMPLA_ANY_H

#include <experimental/any>
#include "Log.h"

namespace simpla {
namespace toolbox {
typedef std::experimental::any Any;
template <typename U>
U const& AnyCast(Any const& a) {
    ASSERT(!a.empty());
    return *std::experimental::any_cast<U>(&a);
};
template <typename U>
U const* AnyCast(Any const* a) {
    ASSERT(a != nullptr && !a->empty());
    return std::experimental::any_cast<U>(a);
};
template <typename U>
U& AnyCast(Any& a) {
    ASSERT(!a.empty());
    return *std::experimental::any_cast<U>(&a);
};
template <typename U>
U* AnyCast(Any* a) {
    ASSERT(a != nullptr && !a->empty());
    return std::experimental::any_cast<U>(a);
};
template <typename U>
U AnyCast(Any&& a) {
    return std::experimental::any_cast<U>(std::forward<Any>(a));
};
}  // namespace toolbox {
}  // namespace simpla {

#endif  // SIMPLA_ANY_H
