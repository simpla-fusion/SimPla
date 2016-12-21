//
// Created by salmon on 16-6-27.
//

#ifndef SIMPLA_BOXUTILITY_H
#define SIMPLA_BOXUTILITY_H

#include <simpla/SIMPLA_config.h>
#include <tuple>
#include <type_traits>

#include "simpla/algebra/nTuple.h"


namespace simpla { namespace toolbox
{
template<typename T> using Point3 = nTuple<T, 3>;
template<typename T> using Vector3 = nTuple<typename std::make_signed<T>::type, 3>;
template<typename T> using Box3 = std::tuple<Point3<T>, Point3<T>>;

template<typename T> constexpr inline T
size(Box3<T> const &b)
{
    return (std::get<1>(b)[0] - std::get<0>(b)[0]) *
           (std::get<1>(b)[1] - std::get<0>(b)[1]) *
           (std::get<1>(b)[2] - std::get<0>(b)[2]);
}

template<typename T> constexpr inline bool
is_valid(Box3<T> const &b)
{
    return (std::get<1>(b)[0] > std::get<0>(b)[0]) &&
           (std::get<1>(b)[1] > std::get<0>(b)[1]) &&
           (std::get<1>(b)[2] > std::get<0>(b)[2]);
}

template<typename T> constexpr inline bool
is_null(Box3<T> const &b)
{
    return (std::get<1>(b)[0] == std::get<0>(b)[0]) ||
           (std::get<1>(b)[1] == std::get<0>(b)[1]) ||
           (std::get<1>(b)[2] == std::get<0>(b)[2]);
}

template<typename T> constexpr inline bool
is_inside(Box3<T> const &left, Box3<T> const &right)
{
    return (std::get<0>(left)[0] >= std::get<0>(right)[0]) &&
           (std::get<0>(left)[1] >= std::get<0>(right)[1]) &&
           (std::get<0>(left)[2] >= std::get<0>(right)[2]) &&
           (std::get<1>(left)[0] <= std::get<1>(right)[0]) &&
           (std::get<1>(left)[1] <= std::get<1>(right)[1]) &&
           (std::get<1>(left)[2] <= std::get<1>(right)[2]);
}

template<typename T> constexpr inline bool
is_inside(Point3<T> const &p, Box3<T> const &b)
{
    return (p[0] >= std::get<0>(b)[0]) &&
           (p[1] >= std::get<0>(b)[1]) &&
           (p[2] >= std::get<0>(b)[2]) &&
           (p[0] <= std::get<1>(b)[0]) &&
           (p[1] <= std::get<1>(b)[1]) &&
           (p[2] <= std::get<1>(b)[2]);
}

template<typename T> constexpr inline bool
is_same(Box3<T> const &left, Box3<T> const &right)
{
    return (std::get<0>(left)[0] == std::get<0>(right)[0]) &&
           (std::get<0>(left)[1] == std::get<0>(right)[1]) &&
           (std::get<0>(left)[2] == std::get<0>(right)[2]) &&
           (std::get<1>(left)[0] == std::get<1>(right)[0]) &&
           (std::get<1>(left)[1] == std::get<1>(right)[1]) &&
           (std::get<1>(left)[2] == std::get<1>(right)[2]);
}

template<typename T> constexpr inline bool
check_adjoining(Box3<T> const &left, Box3<T> const &right, Point3<T> const &dx, Point3<T> const &L)
{
    return false;
}

template<typename T> constexpr inline bool
check_overlapping(Box3<T> const &left, Box3<T> const &right)
{
    return false;
}

template<typename T> constexpr inline nTuple<typename std::make_unsigned<T>::type, 3>
dimensions(Box3<T> const &b)
{
    return nTuple<typename std::make_unsigned<T>::type, 3>
            {
                    static_cast<typename std::make_unsigned<T>::type>(std::get<1>(b)[0] - std::get<0>(b)[0]),
                    static_cast<typename std::make_unsigned<T>::type>(std::get<1>(b)[1] - std::get<0>(b)[1]),
                    static_cast<typename std::make_unsigned<T>::type>(std::get<1>(b)[2] - std::get<0>(b)[2])
            };
}


template<typename U, typename V> constexpr inline Point3<U>
convert(Point3<V> const &b) { return Point3<U>{static_cast<U>(b[0]), static_cast<U>(b[1]), static_cast<U>(b[2])}; }

template<typename U, typename V> constexpr inline Box3<U>
convert(Box3<V> const &b) { return Box3<U>{convert(std::get<0>(b)), convert(std::get<1>(b))}; }


/**
 * get the length of  Space fill curve
 *  default: as  c-order array
 * @param p
 * @param b
 * @return
 */
template<typename T> constexpr inline T
hash_sfc(Point3<T> const &p, Box3<T> const &b)
{
    return ( /**/ p[2] + std::get<1>(b)[2] - std::get<0>(b)[2] * 2) % (std::get<1>(b)[2] - std::get<0>(b)[2]) +
           ((/**/ p[1] + std::get<1>(b)[1] - std::get<0>(b)[1] * 2) % (std::get<1>(b)[1] - std::get<0>(b)[1]) +
            (/**/ p[0] + std::get<1>(b)[0] - std::get<0>(b)[0] * 2) % (std::get<1>(b)[0] - std::get<0>(b)[0])
            * (std::get<1>(b)[2] - std::get<0>(b)[2])) * ((std::get<1>(b)[1] - std::get<0>(b)[1]));
}

template<typename T> constexpr inline Point3<T>
min(Point3<T> const &left, Point3<T> const &right)
{
    return Point3<T>{
            std::min(left[0], right[0]),
            std::min(left[1], right[1]),
            std::min(left[2], right[2]),
    };
}

template<typename T> constexpr inline Point3<T>
max(Point3<T> const &left, Point3<T> const &right)
{
    return Point3<T>{
            std::max(left[0], right[0]),
            std::max(left[1], right[1]),
            std::max(left[2], right[2]),
    };
}

template<typename T>
constexpr inline Box3<T>
intersection(Box3<T> const &left, Box3<T> const &right)
{
    return Box3<T>{max(std::get<0>(left), std::get<0>(right)), min(std::get<1>(left), std::get<1>(right))};
}

template<typename T>
constexpr inline Box3<T>
union_bounding(Box3<T> const &left, Box3<T> const &right)
{
    return Box3<T> {min(std::get<0>(left), std::get<0>(right)), max(std::get<1>(left), std::get<1>(right))};
}

}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_BOXUTILITY_H
