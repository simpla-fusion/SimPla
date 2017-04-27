/**
 * @file ntuple_ext.h
 *
 * @date 2015-6-10
 * @author salmon
 */

#ifndef CORE_NTUPLE_EXT_H_
#define CORE_NTUPLE_EXT_H_

#include <simpla/concept/CheckConcept.h>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>

#include "simpla/utilities/nTuple.h"

namespace simpla {

typedef nTuple<Real, 3ul> point_type;  //!< DataType of configuration space point (coordinates i.e. (x,y,z) )

typedef nTuple<Real, 3ul> vector_type;

typedef std::tuple<point_type, point_type> box_type;  //! two corner of rectangle (or hexahedron ) , <lower ,upper>

typedef long difference_type;  //!< Data type of the difference between indices,i.e.  s = i - j

typedef nTuple<index_type, 3> index_tuple;
typedef nTuple<size_type, 3> size_tuple;

typedef std::tuple<index_tuple, index_tuple> index_box_type;

// typedef std::complex<Real> Complex;

typedef nTuple<Real, 3> Vec3;

typedef nTuple<Real, 3> CoVec3;

typedef nTuple<Integral, 3> IVec3;

typedef nTuple<Real, 3> RVec3;

// typedef nTuple<Complex, 3> CVec3;

template <typename T>
T vec_dot(nTuple<T, 3> const &l, nTuple<T, 3> const &r) {
    return l[0] * r[0] + l[1] * r[1] + l[2] * r[2];
}

template <typename T>
T vec_dot(nTuple<T, 4> const &l, nTuple<T, 4> const &r) {
    return l[0] * r[0] + l[1] * r[1] + l[2] * r[2] + l[3] * r[3];
}
template <typename T, int N>
T vec_dot(nTuple<T, N> const &l, nTuple<T, N> const &r) {
    T res = l[0] * r[0];
    for (int i = 1; i < N; ++i) { res += l[i] * r[i]; }
    return res;
}

template <typename T>
inline T determinant(nTuple<T, 3> const &m) {
    return m[0] * m[1] * m[2];
}

template <typename T>
inline T determinant(nTuple<T, 4> const &m) {
    return m[0] * m[1] * m[2] * m[3];
}

template <typename T>
inline T determinant(nTuple<T, 3, 3> const &m) {
    return m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] * m[1][2] * m[2][0] -
           m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1] * m[0][2] - m[1][2] * m[2][1] * m[0][0];
}

template <typename T>
inline T determinant(nTuple<T, 4, 4> const &m) {
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
// inline nTuple<std::result_of_t<tags::multiplies::eval(traits::value_type_t < T1 > ,
//                                                                traits::value_type_t < T2 > )>, 3>
// cross(T1 const &l, T2 const &r, ENABLE_IF(traits::is_nTuple<T1>::value &&
// traits::is_nTuple<T2>::value))
//{
//    return nTuple<std::result_of_t<tags::multiplies::eval(traits::value_type_t<T1>,
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
//
// template <typename T, int... N>
// auto mod(nTuple<T, N...> const &l) {
//    return std::sqrt(std::abs(inner_product(l, l)));
//}
//
// template <typename TOP, typename T>
// T reduce(T const &v, ENABLE_IF(traits::is_scalar<T>::value)) {
//    return v;
//}
//
// template <typename TOP, typename T>
// traits::value_type_t<T> reduce(T const &v, ENABLE_IF(traits::is_nTuple<T>::value)) {
//    traits::value_type_t<T> res;
//
//    //    static constexpr int n = N0;
//    //
//    //    traits::value_type_t<nTuple<T, N0, N...> > res = reduce(op, traits::get_v(v, 0));
//    //
//    //    for (int s = 1; s < n; ++s) { res = TOP::eval(res, reduce(op, traits::get_v(v, s)));
//    //    }
//
//    return res;
//}
//
// template <typename TL, typename TR>
// inline auto inner_product(TL const &l, TR const &r,
//                          ENABLE_IF(traits::is_nTuple<TL>::value &&traits::is_nTuple<TL>::value)) {
//    return ((reduce<tags::plus>(l * r)));
//}
//
// template <typename T>
// inline auto normal(T const &l, ENABLE_IF(traits::is_nTuple<T>::value)) {
//    return ((std::sqrt(inner_product(l, l))));
//}
//
// template <typename T>
// inline auto abs(T const &l, ENABLE_IF(traits::is_nTuple<T>::value)) {
//    return ((std::sqrt(inner_product(l, l))));
//}
//
// template <typename T>
// inline auto NProduct(T const &v, ENABLE_IF(traits::is_nTuple<T>::value)) {
//    return ((reduce<tags::multiplies>(v)));
//}
//
// template <typename T>
// inline auto NSum(T const &v, ENABLE_IF(traits::is_nTuple<T>::value)) {
//    return ((reduce<tags::plus>(v)));
//}

//
// template<typename T, int N0> std::istream &
// input(std::istream &is, nTuple <T, N0> &tv)
//{
//    for (int i = 0; i < N0 && is; ++i) { is >> tv[i]; }
//    return (is);
//}
//
// template<typename T, int N0, int ...N> std::istream &
// input(std::istream &is, nTuple<T, N0, N ...> &tv)
//{
//    for (int i = 0; i < N0 && is; ++i) { input(is, tv[i]); }
//    return (is);
//}

namespace _detail {
template <typename T, int... N>
std::ostream &printNd_(std::ostream &os, T const &d, int_sequence<N...> const &,
                       ENABLE_IF((!simpla::concept::is_indexable<T>::value))) {
    os << d;
    return os;
}

template <typename T, int M, int... N>
std::ostream &printNd_(std::ostream &os, T const &d, int_sequence<M, N...> const &,
                       ENABLE_IF((simpla::concept::is_indexable<T>::value))) {
    os << "[";
    printNd_(os, d[0], int_sequence<N...>());
    for (int i = 1; i < M; ++i) {
        os << " , ";
        printNd_(os, d[i], int_sequence<N...>());
    }
    os << "]";

    return os;
}

template <typename T>
std::istream &input(std::istream &is, T &a) {
    is >> a;
    return is;
}

template <typename T, int M0, int... M>
std::istream &input(std::istream &is, nTuple<T, M0, M...> &a) {
    for (int n = 0; n < M0; ++n) { _detail::input(is, a[n]); }
    return is;
}

}  // namespace _detail

template <typename T, int... M>
std::ostream &operator<<(std::ostream &os, nTuple<T, M...> const &v) {
    return _detail::printNd_(os, v.data_, int_sequence<M...>());
}

template <typename T, int... M>
std::istream &operator>>(std::istream &is, nTuple<T, M...> &a) {
    _detail::input(is, a);
    return is;
}
template <typename T, int... M>
std::ostream &operator<<(std::ostream &os, std::tuple<nTuple<T, M...>, nTuple<T, M...>> const &v) {
    os << "{ " << std::get<0>(v) << " ," << std::get<1>(v) << "}";
    return os;
};

// template <typename T, int... M>
// std::ostream &operator<<(std::ostream &os, nTuple<T, M...> const &v) {
//    return _detail::printNd_(os, v.data_, int_sequence<M...>());
//}

// namespace traits
//{
// template<typename TV, int...N>
// struct type_cast<algebra::nTuple<TV, N...>, any>
//{
//    typedef algebra::nTuple<TV, N...> TSrc;
//
//    static constexpr any eval(utilities::LuaObject const &v)
//    {
//        return v.as<TDest>();
//    }
//
//};
//}//namespace traits

}  // namespace simpla

#endif /* CORE_NTUPLE_EXT_H_ */
