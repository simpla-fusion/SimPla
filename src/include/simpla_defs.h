/*
 * simpla_defs.h
 *
 *  Created on: 2011-12-24
 *      Author: salmon
 */

#ifndef SIMPLA_DEFS_H_
#define SIMPLA_DEFS_H_

#include <cstddef>
#include <cassert>
#include <limits>
#include <complex>
#include <stdint.h>
#include <cassert>

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
enum POSITION
{
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

static const Real INIFITY = std::numeric_limits<Real>::infinity();
static const Real EPSILON = std::numeric_limits<Real>::epsilon();

class NullType;
class EmptyType
{
};
template<int N> struct Int2Type
{
	static const int value = N;
};

} //namespace simpla

//#ifndef DONOT_USE_TR1

#include <tr1/memory>
#include <tr1/functional>
namespace TR1 = std::tr1;

#define PARALL_FOR( _IT_,_IT_BEGIN_,_IT_END_)                                                            \
			int _ib = _IT_BEGIN_ + (_IT_END_ - _IT_BEGIN_) * omp_thread_num() / omp_num_threads();       \
			int _ie =  _IT_BEGIN_ + (_IT_END_ - _IT_BEGIN_) * (omp_thread_num()+1) / omp_num_threads();  \
			for (int _IT_ = _ib;_IT_ < _ie; ++_IT_)

//#else
//#	include <boost/shared_ptr.hpp>
//#	include <boost/function.hpp>
//#	include <boost/bind.hpp>
//namespace TR1 = boost;
//#endif //DONOT_USE_TR1
//#endif //__GXX_EXPERIMENTAL_CXX0X__

#include "utilities/log.h"


#ifdef NDEBUG
#  define ASSERT(_EXP_)
#else
#  define ASSERT(_COND_)    assert(_COND_);
#endif

#endif /* SIMPLA_DEFS_H_ */
