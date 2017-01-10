/**
 * @file ntuple_ext.h
 *
 * @date 2015-6-10
 * @author salmon
 */

#ifndef CORE_NTUPLE_EXT_H_
#define CORE_NTUPLE_EXT_H_

#include <simpla/mpl/any.h>
#include <simpla/mpl/check_concept.h>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>

#include "nTuple.h"

namespace simpla {
namespace algebra {

template <typename T>
inline T determinant(declare::nTuple_<T, 3> const &m) {
    return m[0] * m[1] * m[2];
}

template <typename T>
inline T determinant(declare::nTuple_<T, 4> const &m) {
    return m[0] * m[1] * m[2] * m[3];
}

template <typename T>
inline T determinant(declare::nTuple_<T, 3, 3> const &m) {
    return m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] * m[1][2] * m[2][0] -
           m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1] * m[0][2] - m[1][2] * m[2][1] * m[0][0];
}

template <typename T>
inline T determinant(declare::nTuple_<T, 4, 4> const &m) {
    return m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0] -
           m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3] * m[2][2] * m[3][0] +
           m[0][2] * m[1][1] * m[2][3] * m[3][0] - m[0][1] * m[1][2] * m[2][3] * m[3][0] -
           m[0][3] * m[1][2] * m[2][0] * m[3][1] + m[0][2] * m[1][3] * m[2][0] * m[3][1] +
           m[0][3] * m[1][0] * m[2][2] * m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1] -
           m[0][2] * m[1][0] * m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1] +
           m[0][3] * m[1][1] * m[2][0] * m[3][2] - m[0][1] * m[1][3] * m[2][0] * m[3][2] -
           m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1] * m[3][2] +
           m[0][1] * m[1][0] * m[2][3] * m[3][2] - m[0][0] * m[1][1] * m[2][3] * m[3][2] -
           m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1] * m[1][2] * m[2][0] * m[3][3] +
           m[0][2] * m[1][0] * m[2][1] * m[3][3] - m[0][0] * m[1][2] * m[2][1] * m[3][3] -
           m[0][1] * m[1][0] * m[2][2] * m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3];
}

// template<typename T1, typename T2>
// inline declare::nTuple_<std::result_of_t<tags::multiplies::eval(traits::value_type_t < T1 > ,
//                                                                traits::value_type_t < T2 > )>, 3>
// cross(T1 const &l, T2 const &r, ENABLE_IF(traits::is_nTuple<T1>::value &&
// traits::is_nTuple<T2>::value))
//{
//    return declare::nTuple_<std::result_of_t<tags::multiplies::eval(traits::value_type_t<T1>,
//                                                                    traits::value_type_t<T2>)>, 3>
//            {
//                    traits::get_v(l, 1) * traits::get_v(r, 2) -
//                    traits::get_v(l, 2) * traits::get_v(r, 1),
//                    traits::get_v(l, 2) * traits::get_v(r, 0) -
//                    traits::get_v(l, 0) * traits::get_v(r, 2),
//                    traits::get_v(l, 0) * traits::get_v(r, 1) -
//                    traits::get_v(l, 1) * traits::get_v(r, 0)
//            };
//}

template <typename T, int... N>
auto mod(nTuple<T, N...> const &l) {
    return std::sqrt(std::abs(inner_product(l, l)));
}

template <typename TOP, typename T>
T reduce(T const &v, ENABLE_IF(traits::is_scalar<T>::value)) {
    return v;
}

template <typename TOP, typename T>
traits::value_type_t<T> reduce(T const &v, ENABLE_IF(traits::is_nTuple<T>::value)) {
    traits::value_type_t<T> res;

    //    static constexpr int n = N0;
    //
    //    traits::value_type_t<nTuple<T, N0, N...> > res = reduce(op, traits::get_v(v, 0));
    //
    //    for (int s = 1; s < n; ++s) { res = TOP::eval(res, reduce(op, traits::get_v(v, s)));
    //    }

    return res;
}

template <typename TL, typename TR>
inline auto inner_product(TL const &l, TR const &r,
                          ENABLE_IF(traits::is_nTuple<TL>::value &&traits::is_nTuple<TL>::value)) {
    return ((reduce<tags::plus>(l * r)));
}

template <typename T>
inline auto normal(T const &l, ENABLE_IF(traits::is_nTuple<T>::value)) {
    return ((std::sqrt(inner_product(l, l))));
}

template <typename T>
inline auto abs(T const &l, ENABLE_IF(traits::is_nTuple<T>::value)) {
    return ((std::sqrt(inner_product(l, l))));
}

template <typename T>
inline auto NProduct(T const &v, ENABLE_IF(traits::is_nTuple<T>::value)) {
    return ((reduce<tags::multiplies>(v)));
}

template <typename T>
inline auto NSum(T const &v, ENABLE_IF(traits::is_nTuple<T>::value)) {
    return ((reduce<tags::plus>(v)));
}

//
// template<typename T, int N0> std::istream &
// input(std::istream &is, declare::nTuple_ <T, N0> &tv)
//{
//    for (int i = 0; i < N0 && is; ++i) { is >> tv[i]; }
//    return (is);
//}
//
// template<typename T, int N0, int ...N> std::istream &
// input(std::istream &is, declare::nTuple_<T, N0, N ...> &tv)
//{
//    for (int i = 0; i < N0 && is; ++i) { input(is, tv[i]); }
//    return (is);
//}

namespace _detail {
template <typename T, int... N>
std::ostream &printNd_(std::ostream &os, T const &d, index_sequence<N...> const &,
                       ENABLE_IF((!simpla::traits::is_indexable<T, int>::value))) {
    os << d;
    return os;
}

template <typename T, int M, int... N>
std::ostream &printNd_(std::ostream &os, T const &d, index_sequence<M, N...> const &,
                       ENABLE_IF((simpla::traits::is_indexable<T, int>::value))) {
    os << "{";
    printNd_(os, d[0], index_sequence<N...>());
    for (int i = 1; i < M; ++i) {
        os << " , ";
        printNd_(os, d[i], index_sequence<N...>());
    }
    os << "}";

    return os;
}

template <typename T, int... M>
std::istream &input(std::istream &is, declare::nTuple_<T, M...> &a) {
    //    _detail::input(is, a);
    return is;
}

}  // namespace _detail
namespace declare {

template <typename T, int... M>
std::ostream &operator<<(std::ostream &os, nTuple_<T, M...> const &v) {
    return _detail::printNd_(os, v.data_, index_sequence<M...>());
}

template <typename T, int... M>
std::istream &operator>>(std::istream &is, nTuple_<T, M...> &a) {
    _detail::input(is, a);
    return is;
}
}  // namespace declare

// template <typename T, int... M>
// std::ostream &operator<<(std::ostream &os, declare::nTuple_<T, M...> const &v) {
//    return _detail::printNd_(os, v.data_, index_sequence<M...>());
//}
}  // namespace algebra

// namespace traits
//{
// template<typename TV, int...N>
// struct type_cast<algebra::declare::nTuple_<TV, N...>, any>
//{
//    typedef algebra::declare::nTuple_<TV, N...> TSrc;
//
//    static constexpr any eval(toolbox::LuaObject const &v)
//    {
//        return v.as<TDest>();
//    }
//
//};
//}//namespace traits

}  // namespace simpla::algebra

#endif /* CORE_NTUPLE_EXT_H_ */
