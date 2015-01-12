/*
 * primitives.h
 *
 *  created on: 2013-6-24
 *      Author: salmon
 */

#ifndef PRIMITIVES_H_
#define PRIMITIVES_H_

#include <sys/types.h>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <valarray>
#include "type_traits.h"
#include "complex.h"
#include "ntuple.h"
namespace simpla
{

enum POSITION
{
	/*
	 FULL = -1, // 11111111
	 CENTER = 0, // 00000000
	 LEFT = 1, // 00000001
	 RIGHT = 2, // 00000010
	 DOWN = 4, // 00000100
	 UP = 8, // 00001000
	 BACK = 16, // 00010000
	 FRONT = 32 //00100000
	 */
	FULL = -1,
	CENTER = 0,
	LEFT = 1,
	RIGHT = 2,
	DOWN = 4,
	UP = 8,
	BACK = 16,
	FRONT = 32
};

enum ArrayOrder
{
	C_ORDER, // SLOW FIRST
	FORTRAN_ORDER //  FAST_FIRST
};

enum GeometryFormTypeID
{
	VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3
};

typedef int8_t ByteType; // int8_t

typedef double Real;

typedef long Integral;

typedef std::complex<Real> Complex;

typedef nTuple<Real, 3> Vec3;

typedef nTuple<Integral, 3> IVec3;

typedef nTuple<Real, 3> RVec3;

typedef nTuple<Complex, 3> CVec3;

static constexpr Real INIFITY = std::numeric_limits<Real>::infinity();

static constexpr Real EPSILON = std::numeric_limits<Real>::epsilon();

static constexpr unsigned int MAX_NDIMS_OF_ARRAY = 10;

static constexpr unsigned int CARTESIAN_XAXIS = 0;
static constexpr unsigned int CARTESIAN_YAXIS = 1;
static constexpr unsigned int CARTESIAN_ZAXIS = 2;

template<typename > struct has_PlaceHolder
{
	static constexpr bool value = false;
};

template<typename > struct is_real
{
	static constexpr bool value = false;
};

template<> struct is_real<Real>
{
	static constexpr bool value = true;
};

template<typename TL>
struct is_arithmetic_scalar
{
	static constexpr bool value = (std::is_arithmetic<TL>::value
			|| has_PlaceHolder<TL>::value);
};

template<typename T>
struct is_primitive
{
	static constexpr bool value = is_arithmetic_scalar<T>::value;
};

template<typename T>
struct is_expression
{
	static constexpr bool value = false;
};

template<typename T1> auto abs(T1 const & m)
DECL_RET_TYPE ((std::fabs(m)))

//
//template<typename T>
//auto Reciprocal(T const & f)
//DECL_RET_TYPE(( 1.0/f))
//
//
//namespace ops
//{
//
//	template<typename TL, typename TI>
//	constexpr auto get_index(TL const & l, TI s)->TL const&
//	{
//		return l;
//	}
//	template<typename TL, typename TI>
//	constexpr auto get_index(TL const * l, TI s)->TL const&
//	{
//		return l[s];
//	}
//	;
//	template<  unsigned int    N, typename TL, typename TI>
//	constexpr auto get_index(nTuple<N, TL> const & l, TI s)->TL const&
//	{
//		return l[s];
//	}
//
////ENABLE_IF_DECL_RET_TYPE(is_indexable<TL>::value ,l[s])
//
////template<typename TL, typename TI>
////  auto get_index(TL const & l, TI s)
////ENABLE_IF_DECL_RET_TYPE(!is_indexable<TL>::value ,l)
//
//	template<typename TOP, typename TL, typename TR, typename TI>
//	auto calculus(TOP op, TL const & l, TR const &r, TI s)
//	DECL_RET_TYPE(op(get_index(l,s),get_index(r,s)))
//
//	template<typename TOP, typename TL, typename TI>
//	auto calculus(TOP, TL const & l, TI s)
//	DECL_RET_TYPE(op(get_index(l,s) ))
//
//}  // namespace ops

}
// namespace simpla
#endif /* PRIMITIVES_H_ */
