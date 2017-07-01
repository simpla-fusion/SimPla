/**
 * @file integer_sequence.h
 *
 *  Created on: 2014-9-26
 *      Author: salmon
 */

#ifndef CORE_toolbox_INTEGER_SEQUENCE_H_
#define CORE_toolbox_INTEGER_SEQUENCE_H_

#include <simpla/engine/SPObject.h>
#include <stddef.h>
#include <iostream>
#include "simpla/concept/CheckConcept.h"

namespace simpla {
using namespace concept;
//////////////////////////////////////////////////////////////////////
/// integer_sequence
//////////////////////////////////////////////////////////////////////

//**************************************************************************

//#endif
namespace tags {
template <int V0, int V1, int V2>
using VERSION = std::index_sequence<V0, V1, V2>;
}

//
// template <typename _Tp, _Tp First, _Tp Second, _Tp... Others>
// std::ostream &operator<<(std::ostream &os, std::index_sequence<_Tp, First, Second, Others...> const &) {
//    os << First << " , " <<
//        std::index_sequence<_Tp, Second, Others...>();
//
//    return os;
//}
//
// template <typename _Tp, _Tp First>
// std::ostream &operator<<(std::ostream &os, std::index_sequence<_Tp, First> const &) {
//    os << First;
//    return os;
//}
//
// template <typename _Tp>
// std::ostream &operator<<(std::ostream &os, std::index_sequence<_Tp> const &) {
//    return os;
//}
//
// template <int I>
// using int_const = std::integral_constant<int, I>;
//
// template <typename _Tp, _Tp I>
// using integral_constant = std::integral_constant<_Tp, I>;
//
// static const std::index_sequence<int, 0> _0{};
// static const std::index_sequence<int, 1> _1{};
// static const std::index_sequence<int, 2> _2{};
// static const std::index_sequence<int, 3> _3{};
// static const std::index_sequence<int, 4> _4{};
// static const std::index_sequence<int, 5> _5{};
// static const std::index_sequence<int, 6> _6{};
// static const std::index_sequence<int, 7> _7{};
// static const std::index_sequence<int, 8> _8{};
// static const std::index_sequence<int, 9> _9{};
//
// template <typename _T1>
// auto operator-(std::index_sequence<_T1>) {
//    return integer_sequence<_T1>();
//}
//
// template <typename _T1, _T1 I0, _T1... I>
// auto operator-(integer_sequence<_T1, I0, I...>) {
//    return integer_sequence<_T1, -I0>(), (-integer_sequence<_T1, I...>());
//}
//
// template <typename _T1, _T1... I, typename _T2>
// auto operator+(integer_sequence<_T1, I...>, integer_sequence<_T2>) {
//    return integer_sequence<_T1, I...>();
//}
//
// template <typename _T1, typename _T2, _T2... J>
// auto operator+(integer_sequence<_T1>, integer_sequence<_T2, J...>) {
//    return integer_sequence<_T2, J...>();
//}
// template <typename _T1>
// auto operator+(integer_sequence<_T1>, integer_sequence<_T1>) {
//    return integer_sequence<_T1>();
//}
//
// template <typename _T1, _T1 I0, _T1... I, typename _T2, _T2 J0, _T2... J>
// auto operator+(integer_sequence<_T1, I0, I...>, integer_sequence<_T2, J0, J...>) {
//    return integral_constant<_T1, (I0 + J0)>(), (integer_sequence<_T1, I...>() + integer_sequence<_T2, J...>());
//}
//
// template <typename _T1, _T1... I, typename _T2>
// auto operator-(integer_sequence<_T1, I...>, integer_sequence<_T2>) {
//    return integer_sequence<_T1, I...>();
//}
//
// template <typename _T1, typename _T2, _T2... J>
// auto operator-(integer_sequence<_T1>, integer_sequence<_T2, J...>) {
//    return -integer_sequence<_T2, J...>();
//}
// template <typename _T1>
// auto operator-(integer_sequence<_T1>, integer_sequence<_T1>) {
//    return integer_sequence<_T1>();
//}
//
// template <typename _T1, _T1 I0, _T1... I, typename _T2, _T2 J0, _T2... J>
// auto operator-(integer_sequence<_T1, I0, I...>, integer_sequence<_T2, J0, J...>) {
//    return integral_constant<_T1, (I0 - J0)>(), (integer_sequence<_T1, I...>() - integer_sequence<_T2, J...>());
//}
//
// template <typename _T1, _T1... I, typename _T2, _T2 M>
// auto operator*(integer_sequence<_T1, I...>, integral_constant<_T2, M>) {
//    return integer_sequence<_T1, I * M...>();
//}
//
// template <typename _T1, _T1... I, typename _T2, _T2... J>
// auto operator,(integer_sequence<_T1, I...>, integer_sequence<_T2, J...>) {
//    return integer_sequence<_T1, I..., J...>();
//}
template <typename _Tp, _Tp... N>
struct seq_max;
template <typename _Tp, _Tp N0>
struct seq_max<_Tp, N0> : public std::integral_constant<_Tp, N0> {};

template <typename _Tp, _Tp N0, _Tp N1>
struct seq_max<_Tp, N0, N1> : public std::integral_constant<_Tp, (N0 > N1 ? N0 : N1)> {};

template <typename _Tp, _Tp N0, _Tp... N>
struct seq_max<_Tp, N0, N...> : public std::integral_constant<_Tp, seq_max<_Tp, N0, seq_max<_Tp, N...>::value>::value> {
};

template <typename _Tp, _Tp... N>
struct seq_min;

template <typename _Tp, _Tp N0, _Tp... N>
struct seq_min<_Tp, N0, N...> : public std::integral_constant<_Tp, seq_min<_Tp, N0, seq_min<_Tp, N...>::value>::value> {
};

template <typename _Tp, _Tp N0>
struct seq_min<_Tp, N0> : public std::integral_constant<_Tp, N0> {};

template <typename _Tp, _Tp N0, _Tp N1>
struct seq_min<_Tp, N0, N1> : public std::integral_constant<_Tp, (N0 < N1 ? N0 : N1)> {};

}  // namespace simpla
#endif /* CORE_toolbox_INTEGER_SEQUENCE_H_ */
