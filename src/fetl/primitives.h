/*
 * primitives.h
 *
 *  Created on: 2013-6-24
 *      Author: salmon
 */

#ifndef PRIMITIVES_H_
#define PRIMITIVES_H_

#include <sys/types.h>
#include <complex>
#include <limits>
#include "../utilities/type_utilites.h"

namespace simpla
{

template<int N> struct Int2Type
{
	static const int value = N;
};

struct NullType;

struct EmptyType
{
};

enum CONST_NUMBER
{
	ZERO = 0, ONE = 1, TWO = 2, THREE = 3, FOUR = 4, FIVE = 5, SIX = 6, SEVEN = 7, EIGHT = 8, NINE = 9
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
	FULL = -1, CENTER = 0, LEFT = 1, RIGHT = 2, DOWN = 4, UP = 8, BACK = 16, FRONT = 32
};

typedef int8_t ByteType; // int8_t

typedef double Real;

typedef long Integral;

typedef std::complex<Real> Complex;

template<int N, typename T> struct nTuple;

template<typename, int> struct Geometry;

template<typename, typename > struct Field;

template<int TOP, typename TL, typename TR> struct BiOp;

template<int TOP, typename TL> struct UniOp;

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

template<typename T>
struct remove_const_reference
{
	typedef typename std::remove_const<typename std::remove_reference<T>::type>::type type;
};

template<typename > struct is_complex
{
	static const bool value = false;
};

template<typename T> struct is_complex<std::complex<T> >
{
	static const bool value = true;
};

template<int TOP, typename TL, typename TR> struct is_complex<BiOp<TOP, TL, TR> >
{
	static const bool value = is_complex<TL>::value || is_complex<TR>::value;
};

template<typename > struct is_real
{
	static const bool value = false;
};

template<> struct is_real<Real>
{
	static const bool value = true;
};

template<int TOP, typename TL, typename TR> struct is_real<BiOp<TOP, TL, TR> >
{
	static const bool value = is_real<TL>::value && is_real<TR>::value;
};

template<typename > struct has_PlaceHolder
{
	static const bool value = false;
};

template<typename TL>
struct is_arithmetic_scalar
{
	static const bool value = (std::is_arithmetic<TL>::value || is_complex<TL>::value || has_PlaceHolder<TL>::value);
};

template<typename T>
struct is_primitive
{
	static const bool value = is_arithmetic_scalar<T>::value;
};

template<int N, typename TE>
struct is_primitive<nTuple<N, TE> >
{
	static const bool value = is_arithmetic_scalar<TE>::value;
};

template<typename T>
struct is_storage_type
{
	static const bool value = false;
};
template<typename T>
struct is_storage_type<std::complex<T>>
{
	static const bool value = false;
};

template<typename TG, typename T>
struct is_storage_type<Field<TG, T> >
{
	static const bool value = true;
};

template<typename TG, int TOP, typename TL, typename TR>
struct is_storage_type<Field<TG, BiOp<TOP, TL, TR> > >
{
	static const bool value = false;
};

template<typename TG, int TOP, typename TL>
struct is_storage_type<Field<TG, UniOp<TOP, TL> > >
{
	static const bool value = false;
};

template<typename T>
struct is_ntuple
{
	static const bool value = false;
};

template<int N, typename T>
struct is_ntuple<nTuple<N, T>>
{
	static const bool value = true;
};

template<typename T>
struct ReferenceTraits
{
	typedef typename remove_const_reference<T>::type TL;
	typedef typename std::conditional<is_storage_type<TL>::value, TL &, TL>::type type;
};

template<typename T>
struct ConstReferenceTraits
{
	typedef typename remove_const_reference<T>::type TL;
	typedef typename std::conditional<is_storage_type<TL>::value, TL const &, const TL>::type type;
};

//template<class T, typename TI = int>
//struct is_indexable
//{
//	template<typename T1, typename T2>
//	static auto check_index(T1 const& u, T2 const &s) ->typename std::add_const<
//	decltype(const_cast<typename std::remove_cv<T1>::type &>(u)[s])>::type
//	{
//	}
//
//	static std::false_type check_index(...)
//	{
//		return std::false_type();
//	}
//
//public:
//
//	typedef decltype(
//			check_index((std::declval<T>()),
//					std::declval<TI>())) result_type;
//
//	static const bool value = !(std::is_same<result_type, std::false_type>::value);
//
//};

template<class T, typename TI = int>
class is_indexable
{
	HAS_OPERATOR(index, []);

public:

	static const bool value = has_operator_index<T, TI>::value;

};

typedef enum
{
	PLUS = 1, MINUS = 2, MULTIPLIES = 3, DIVIDES = 4, NEGATE = 5,

	MODULUS, BITWISEXOR, BITWISEAND, BITWISEOR,

	// Tensor product
	TENSOR_PRODUCT, // outer product
	TENSOR_CONTRACTION,

	WEDGE,
	HODGESTAR,
	EXTRIORDERIVATIVE,

	CROSS,
	DOT,

	GRAD = 20,
	DIVERGE,
	CURL,
	CURLPDX,
	CURLPDY,
	CURLPDZ,

	MAPTO0,
	MAPTO1,
	MAPTO2,
	MAPTO3,

	EQUAL,
	LESS,
	GREATER,

	NULL_OP

} OpType;

typedef enum
{
	SIN = NULL_OP + 1, COS, TAN, CTAN, EXP, LOG10, LOG2, LN, ABS

} MathFunType;

template<int TOP, typename TL, typename TR> struct OpTraits;
template<typename T>
struct FieldTraits
{
	enum
	{
		is_field = false
	};

	enum
	{
		IForm = 0
	}
	;
	typedef T value_type;
};

template<typename TM, int IFORM, typename TExpr>
struct FieldTraits<Field<Geometry<TM, IFORM>, TExpr> >
{
	typedef Field<Geometry<TM, IFORM>, TExpr> this_type;
	enum
	{
		is_field = true
	};

	enum
	{
		IForm = IFORM
	}
	;
	typedef typename this_type::value_type value_type;
};

template<typename TL>
struct is_field
{
	static const bool value = FieldTraits<TL>::is_field;
};

template<typename T> auto abs(T const & m)
DECL_RET_TYPE((std::abs(m)))

template<typename TV, typename TR> inline TV TypeCast(TR const & obj)
{
	return std::move(static_cast<TV>(obj));
}
}
// namespace simpla
#endif /* PRIMITIVES_H_ */
