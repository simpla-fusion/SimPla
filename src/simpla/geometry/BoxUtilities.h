//
// Created by salmon on 17-7-20.
//

#ifndef SIMPLA_BOXUTILITIES_H
#define SIMPLA_BOXUTILITIES_H

#include "simpla/algebra/nTuple.h"
#include "simpla/utilities/SPDefines.h"

namespace simpla {
template <typename, int...>
struct nTuple;
namespace geometry {

template <typename T>
constexpr inline T Measure(std::tuple<nTuple<T, 2>, nTuple<T, 2>> const &b) {
    return (std::get<1>(b)[0] - std::get<0>(b)[0]) * (std::get<1>(b)[1] - std::get<0>(b)[1]);
}
template <typename T>
constexpr inline T Measure(std::tuple<nTuple<T, 3>, nTuple<T, 3>> const &b) {
    return (std::get<1>(b)[0] - std::get<0>(b)[0]) * (std::get<1>(b)[1] - std::get<0>(b)[1]) *
           (std::get<1>(b)[2] - std::get<0>(b)[2]);
}

template <typename T>
constexpr inline std::tuple<nTuple<T, 2>, nTuple<T, 2>> Overlap(std::tuple<nTuple<T, 2>, nTuple<T, 2>> const &a,
                                                                std::tuple<nTuple<T, 2>, nTuple<T, 2>> const &b) {
    return std::tuple<nTuple<T, 2>, nTuple<T, 2>>{
        {std::max(std::get<0>(a)[0], std::get<0>(b)[0]), std::max(std::get<0>(a)[1], std::get<0>(b)[1])},
        {std::min(std::get<1>(a)[0], std::get<1>(b)[0]), std::min(std::get<1>(a)[1], std::get<1>(b)[1])}};
};

template <typename T>
constexpr inline std::tuple<nTuple<T, 3>, nTuple<T, 3>> Overlap(std::tuple<nTuple<T, 3>, nTuple<T, 3>> const &a,
                                                                std::tuple<nTuple<T, 3>, nTuple<T, 3>> const &b) {
    return std::tuple<nTuple<T, 3>, nTuple<T, 3>>{
        {std::max(std::get<0>(a)[0], std::get<0>(b)[0]), std::max(std::get<0>(a)[1], std::get<0>(b)[1]),
         std::max(std::get<0>(a)[2], std::get<0>(b)[2])},
        {std::min(std::get<1>(a)[0], std::get<1>(b)[0]), std::min(std::get<1>(a)[1], std::get<1>(b)[1]),
         std::min(std::get<1>(a)[2], std::get<1>(b)[2])}};
};

template <typename T>
constexpr inline std::tuple<nTuple<T, 2>, nTuple<T, 2>> Union(std::tuple<nTuple<T, 2>, nTuple<T, 2>> const &a,
                                                              std::tuple<nTuple<T, 2>, nTuple<T, 2>> const &b) {
    return std::tuple<nTuple<T, 2>, nTuple<T, 2>>{
        {std::min(std::get<0>(a)[0], std::get<0>(b)[0]), std::min(std::get<0>(a)[1], std::get<0>(b)[1])},
        {std::max(std::get<1>(a)[0], std::get<1>(b)[0]), std::max(std::get<1>(a)[1], std::get<1>(b)[1])}};
};

template <typename T>
constexpr inline std::tuple<nTuple<T, 3>, nTuple<T, 3>> Union(std::tuple<nTuple<T, 3>, nTuple<T, 3>> const &a,
                                                              std::tuple<nTuple<T, 3>, nTuple<T, 3>> const &b) {
    return std::tuple<nTuple<T, 3>, nTuple<T, 3>>{
        {std::min(std::get<0>(a)[0], std::get<0>(b)[0]), std::min(std::get<0>(a)[1], std::get<0>(b)[1]),
         std::min(std::get<0>(a)[2], std::get<0>(b)[2])},
        {std::max(std::get<1>(a)[0], std::get<1>(b)[0]), std::max(std::get<1>(a)[1], std::get<1>(b)[1]),
         std::max(std::get<1>(a)[2], std::get<1>(b)[2])}};
};

template <typename T, int N>
constexpr inline std::tuple<nTuple<T, N>, nTuple<T, N>> Expand(std::tuple<nTuple<T, N>, nTuple<T, N>> const &b,
                                                               nTuple<T, N> const &gw) {
    nTuple<T, N> lo = std::get<0>(b) - gw;
    nTuple<T, N> hi = std::get<1>(b) + gw;
    return std::make_tuple(lo, hi);
};

template <typename T>
constexpr inline bool CheckInSide(std::tuple<nTuple<T, 3>, nTuple<T, 3>> const &b, nTuple<T, 3> const &p) {
    return (p[0] >= std::get<0>(b)[0]) && (p[1] >= std::get<0>(b)[1]) && (p[2] >= std::get<0>(b)[2]) &&
           (p[0] <= std::get<1>(b)[0]) && (p[1] <= std::get<1>(b)[1]) && (p[2] <= std::get<1>(b)[2]);
}

//
// template <typename T>
// constexpr inline T size(Box3<T> const &b) {
//    return (std::get<1>(b)[0] - std::get<0>(b)[0]) * (std::get<1>(b)[1] - std::get<0>(b)[1]) *
//           (std::get<1>(b)[2] - std::get<0>(b)[2]);
//}
//
// template <typename T>
// constexpr inline bool is_valid(Box3<T> const &b) {
//    return (std::get<1>(b)[0] > std::get<0>(b)[0]) && (std::get<1>(b)[1] > std::get<0>(b)[1]) &&
//           (std::get<1>(b)[2] > std::get<0>(b)[2]);
//}
//
// template <typename T>
// constexpr inline bool is_null(Box3<T> const &b) {
//    return (std::get<1>(b)[0] == std::get<0>(b)[0]) || (std::get<1>(b)[1] == std::get<0>(b)[1]) ||
//           (std::get<1>(b)[2] == std::get<0>(b)[2]);
//}
//
// template <typename T>
// constexpr inline bool is_inside(Box3<T> const &left, Box3<T> const &right) {
//    return (std::get<0>(left)[0] >= std::get<0>(right)[0]) && (std::get<0>(left)[1] >= std::get<0>(right)[1]) &&
//           (std::get<0>(left)[2] >= std::get<0>(right)[2]) && (std::get<1>(left)[0] <= std::get<1>(right)[0]) &&
//           (std::get<1>(left)[1] <= std::get<1>(right)[1]) && (std::get<1>(left)[2] <= std::get<1>(right)[2]);
//}
//

//
// template <typename T>
// constexpr inline bool is_same(Box3<T> const &left, Box3<T> const &right) {
//    return (std::get<0>(left)[0] == std::get<0>(right)[0]) && (std::get<0>(left)[1] == std::get<0>(right)[1]) &&
//           (std::get<0>(left)[2] == std::get<0>(right)[2]) && (std::get<1>(left)[0] == std::get<1>(right)[0]) &&
//           (std::get<1>(left)[1] == std::get<1>(right)[1]) && (std::get<1>(left)[2] == std::get<1>(right)[2]);
//}
//
// template <typename T>
// constexpr inline bool check_adjoining(Box3<T> const &left, Box3<T> const &right, Point3<T> const &dx,
//                                      Point3<T> const &L) {
//    return false;
//}
//
// template <typename T>
// constexpr inline bool check_overlapping(Box3<T> const &left, Box3<T> const &right) {
//    return false;
//}
//
// template <typename T>
// constexpr inline nTuple<typename std::make_unsigned<T>::type, 3> dimensions(Box3<T> const &b) {
//    return nTuple<typename std::make_unsigned<T>::type, 3>{
//        static_cast<typename std::make_unsigned<T>::type>(std::get<1>(b)[0] - std::get<0>(b)[0]),
//        static_cast<typename std::make_unsigned<T>::type>(std::get<1>(b)[1] - std::get<0>(b)[1]),
//        static_cast<typename std::make_unsigned<T>::type>(std::get<1>(b)[2] - std::get<0>(b)[2])};
//}
//
// template <typename U, typename V>
// constexpr inline Point3<U> convert(Point3<V> const &b) {
//    return Point3<U>{static_cast<U>(b[0]), static_cast<U>(b[1]), static_cast<U>(b[2])};
//}
//
// template <typename U, typename V>
// constexpr inline Box3<U> convert(Box3<V> const &b) {
//    return Box3<U>{convert(std::get<0>(b)), convert(std::get<1>(b))};
//}
//
///**
// * get the length of  Space fill curve
// *  default: as  c-order array
// * @param p
// * @param b
// * @return
// */
// template <typename T>
// constexpr inline T hash_sfc(Point3<T> const &p, Box3<T> const &b) {
//    return (/**/ p[2] + std::get<1>(b)[2] - std::get<0>(b)[2] * 2) % (std::get<1>(b)[2] - std::get<0>(b)[2]) +
//           ((/**/ p[1] + std::get<1>(b)[1] - std::get<0>(b)[1] * 2) % (std::get<1>(b)[1] - std::get<0>(b)[1]) +
//            (/**/ p[0] + std::get<1>(b)[0] - std::get<0>(b)[0] * 2) % (std::get<1>(b)[0] - std::get<0>(b)[0]) *
//                (std::get<1>(b)[2] - std::get<0>(b)[2])) *
//               ((std::get<1>(b)[1] - std::get<0>(b)[1]));
//}
//
// template <typename T>
// constexpr inline Point3<T> min(Point3<T> const &left, Point3<T> const &right) {
//    return Point3<T>{
//        std::min(left[0], right[0]), std::min(left[1], right[1]), std::min(left[2], right[2]),
//    };
//}
//
// template <typename T>
// constexpr inline Point3<T> max(Point3<T> const &left, Point3<T> const &right) {
//    return Point3<T>{
//        std::max(left[0], right[0]), std::max(left[1], right[1]), std::max(left[2], right[2]),
//    };
//}
//
// template <typename T>
// constexpr inline Box3<T> intersection(Box3<T> const &left, Box3<T> const &right) {
//    return Box3<T>{max(std::get<0>(left), std::get<0>(right)), min(std::get<1>(left), std::get<1>(right))};
//}
//
// template <typename T>
// constexpr inline Box3<T> union_bounding(Box3<T> const &left, Box3<T> const &right) {
//    return Box3<T>{min(std::get<0>(left), std::get<0>(right)), max(std::get<1>(left), std::get<1>(right))};
//}

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BOXUTILITIES_H
