//
// Created by salmon on 17-7-24.
//

#ifndef SIMPLA_SPDEFINES_H
#define SIMPLA_SPDEFINES_H
#include <simpla/SIMPLA_config.h>
#include <limits>
#include <tuple>
#include <utility>
//#include "simpla/algebra/nTuple.h"
namespace simpla {

template <typename T, int...>
class nTuple;
// typedef std::complex<Real> Complex;

template <typename T, int N>
using Vector = nTuple<T, N>;

template <typename T, int M, int N>
using Matrix = nTuple<T, M, N>;

template <typename T, int... N>
using Tensor = nTuple<T, N...>;

typedef nTuple<Real, 3> point_type;  //!< DataType of configuration space point (coordinates i.e. (x,y,z) )
typedef nTuple<Real, 2> point2d_type;
typedef nTuple<Real, 3> vector_type;
typedef nTuple<Real, 2> vector2d_type;
typedef nTuple<Real, 3, 3> matrix_type;

typedef std::tuple<point_type, point_type> box_type;  //! two corner of rectangle (or hexahedron ) , <lower ,upper>
typedef std::tuple<point2d_type, point2d_type> box2d_type;

typedef nTuple<index_type, 3> index_tuple;

typedef nTuple<size_type, 3> size_tuple;

typedef std::tuple<index_tuple, index_tuple> index_box_type;

typedef nTuple<Real, 3> Vec3;

typedef nTuple<Real, 3> CoVec3;

typedef nTuple<Integral, 3> IVec3;

typedef nTuple<Real, 3> RVec3;

namespace utility {
template <typename T>
auto make_point(T const* d) {
    return nTuple<T, 3>{d[0], d[1], d[2]};
}
template <typename T>
auto make_box(T const& d) {
    return std::make_tuple(make_point(d[0]), make_point(d[1]));
}

}  //    namespace utility{

// typedef nTuple<Complex, 3> CVec3;
}  // namespace simpla
#endif  // SIMPLA_SPDEFINES_H
