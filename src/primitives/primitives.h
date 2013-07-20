/*
 * primitives.h
 *
 *  Created on: 2013-6-24
 *      Author: salmon
 */

#ifndef PRIMITIVES_H_
#define PRIMITIVES_H_

#include <complex>

namespace simpla
{

static const int NIL = 0;
enum CONST_NUMBER
{
	ZERO = 0,
	ONE = 1,
	TWO = 2,
	THREE = 3,
	FOUR = 4,
	FIVE = 5,
	SIX = 6,
	SEVEN = 7,
	EIGHT = 8,
	NINE = 9
};

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

typedef int8_t ByteType; // int8_t
typedef double Real;
typedef long Integral;

typedef std::complex<Real> Complex;

template<int N, typename T> struct nTuple;
typedef nTuple<THREE, Real> Vec3;
typedef nTuple<THREE, nTuple<THREE, Real> > Tensor3;
typedef nTuple<FOUR, nTuple<FOUR, Real> > Tensor4;

typedef nTuple<THREE, Integral> IVec3;
typedef nTuple<THREE, Real> RVec3;
typedef nTuple<THREE, Complex> CVec3;

typedef nTuple<THREE, nTuple<THREE, Real> > RTensor3;
typedef nTuple<THREE, nTuple<THREE, Complex> > CTensor3;

typedef nTuple<FOUR, nTuple<FOUR, Real> > RTensor4;
typedef nTuple<FOUR, nTuple<FOUR, Complex> > CTensor4;

static const Real INIFITY = std::numeric_limits<Real>::infinity();
static const Real EPSILON = std::numeric_limits<Real>::epsilon();

template<typename T> inline T Abs(T const & v)
{
	return abs(v);
}

template<typename T> inline T Abs(typename std::complex<T> const & v)
{
	return abs(v);
}
}
// namespace simpla
#include "ntuple.h"
#endif /* PRIMITIVES_H_ */
