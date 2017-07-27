//
// Created by salmon on 17-7-24.
//

#ifndef SIMPLA_SPDEFINES_H
#define SIMPLA_SPDEFINES_H
#include <simpla/SIMPLA_config.h>
#include <limits>
#include "simpla/algebra/nTuple.h"
namespace simpla {

static constexpr Real SNaN = std::numeric_limits<Real>::signaling_NaN();

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

typedef nTuple<Real, 3> vector_type;

typedef std::tuple<point_type, point_type> box_type;  //! two corner of rectangle (or hexahedron ) , <lower ,upper>

typedef nTuple<index_type, 3> index_tuple;

typedef nTuple<size_type, 3> size_tuple;

typedef std::tuple<index_tuple, index_tuple> index_box_type;

typedef nTuple<Real, 3> Vec3;

typedef nTuple<Real, 3> CoVec3;

typedef nTuple<Integral, 3> IVec3;

typedef nTuple<Real, 3> RVec3;

// typedef nTuple<Complex, 3> CVec3;
}  // namespace simpla
#endif  // SIMPLA_SPDEFINES_H
