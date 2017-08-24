//
// Created by salmon on 17-3-8.
//

#ifndef SIMPLA_DATAENTITYFACTROY_H
#define SIMPLA_DATAENTITYFACTROY_H

#include <memory>
#include <string>
#include "simpla/algebra/nTuple.h"

namespace simpla {
template <typename TV, int N0, int... N>
struct nTuple<TV, N0, N...>;
namespace data {
class DataEntity;
}  // namespace data {

namespace traits {
template <typename U>
struct is_light_data : public std::integral_constant<bool, std::is_arithmetic<U>::value> {};

// template <typename U, int... N>
// struct is_light_data<nTuple<U, N...>> : public std::true_type {};
// template <typename... U>
// struct is_light_data<std::tuple<U...>> : public std::true_type {};
template <>
struct is_light_data<std::string> : public std::integral_constant<bool, true> {};
template <>
struct is_light_data<char> : public std::integral_constant<bool, true> {};
template <>
struct is_light_data<bool> : public std::integral_constant<bool, true> {};
}  // namespace traits {

}  // namespace simpla {
#endif  // SIMPLA_DATAENTITYFACTROY_H
