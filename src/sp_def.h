//
// Created by salmon on 16-6-27.
//

#ifndef SIMPLA_SP_DEF_H
#define SIMPLA_SP_DEF_H

#include "SIMPLA_config.h"
#include "toolbox/nTuple.h"
#include <boost/uuid/uuid.hpp>

namespace simpla
{

typedef nTuple<Real, 3ul> point_type; //!< DataType of configuration space point (coordinates i.e. (x,y,z) )

typedef nTuple<Real, 3ul> vector_type;

typedef std::tuple<point_type, point_type> box_type; //! two corner of rectangle (or hexahedron ) , <lower ,upper>


typedef long difference_type; //!< Data type of the difference between indices,i.e.  s = i - j

typedef nTuple<size_type, 3> index_tuple;

typedef std::tuple<index_tuple, index_tuple> index_box_type;

typedef boost::uuids::uuid uuid;
/**
 *  @ingroup toolbox
 * @{
 */
//enum POSITION
//{
//	/*
//	 FULL = -1, // 11111111
//	 CENTER = 0, // 00000000
//	 LEFT = 1, // 00000001
//	 RIGHT = 2, // 00000010
//	 DOWN = 4, // 00000100
//	 UP = 8, // 00001000
//	 BACK = 16, // 00010000
//	 FRONT = 32 //00100000
//	 */
//	FULL = -1, //!< FULL
//	CENTER = 0, //!< CENTER
//	LEFT = 1,  //!< LEFT
//	RIGHT = 2, //!< RIGHT
//	DOWN = 4,  //!< DOWN
//	UP = 8,    //!< UP
//	BACK = 16, //!< BACK
//	FRONT = 32 //!< FRONT
//};
//
enum ArrayOrder
{
    C_ORDER, // SLOW FIRST
    FORTRAN_ORDER //  FAST_FIRST
};

typedef Real scalar_type;

typedef std::complex<Real> Complex;

typedef nTuple<Real, 3> Vec3;

typedef nTuple<Real, 3> Covec3;

typedef nTuple<Integral, 3> IVec3;

typedef nTuple<Real, 3> RVec3;

typedef nTuple<Complex, 3> CVec3;

static constexpr Real INIFITY = std::numeric_limits<Real>::infinity();

static constexpr Real EPSILON = std::numeric_limits<Real>::epsilon();

static constexpr unsigned int MAX_NDIMS_OF_ARRAY = 8;

static constexpr unsigned int CARTESIAN_XAXIS = 0;

static constexpr unsigned int CARTESIAN_YAXIS = 1;

static constexpr unsigned int CARTESIAN_ZAXIS = 2;

template<typename>
struct has_PlaceHolder { static constexpr bool value = false; };

template<typename>
struct is_real { static constexpr bool value = false; };

template<>
struct is_real<Real> { static constexpr bool value = true; };

template<typename TL>
struct is_arithmetic_scalar
{
    static constexpr bool value = (std::is_arithmetic<TL>::value || has_PlaceHolder<TL>::value);
};

template<typename T>
struct is_primitive { static constexpr bool value = is_arithmetic_scalar<T>::value; };

//template<typename T> struct is_expression { static constexpr bool entity = false; };

template<typename T1>
auto abs(T1 const &m) DECL_RET_TYPE((std::fabs(m)))

}
#endif //SIMPLA_SP_DEF_H
