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

enum GeometryFormTypeID
{
	VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3
};

typedef int8_t ByteType; // int8_t

typedef double Real;

typedef long Integral;

typedef std::complex<Real> Complex;

template<int N, typename T> struct nTuple;

template<typename T> inline constexpr T real(T const &v)
{
	return v;
}
template<typename T> inline constexpr T imag(T const &)
{
	return 0;
}

template<typename T> inline constexpr T real(std::complex<T> const &v)
{
	return v.real();
}
template<typename T> inline constexpr T imag(std::complex<T> const &v)
{
	return v.imag();
}

template<typename T> inline constexpr nTuple<3, T> real(nTuple<3, std::complex<T>> const &v)
{
	return std::move(nTuple<3, T>( { v[0].real(), v[1].real(), v[2].real() }));
}

template<typename T> inline constexpr nTuple<3, T> imag(nTuple<3, std::complex<T>> const &v)
{
	return std::move(nTuple<3, T>( { v[0].imag(), v[1].imag(), v[2].imag() }));
}

template<int N, typename T> inline nTuple<N, T> real(nTuple<N, std::complex<T>> const &v)
{
	nTuple<N, T> res;
	for (int i = 0; i < N; ++i)
	{
		res[i] = v[i].real();
	}
	return std::move(res);
}

template<int N, typename T> inline nTuple<N, T> imag(nTuple<N, std::complex<T>> const &v)
{
	nTuple<N, T> res;
	for (int i = 0; i < N; ++i)
	{
		res[i] = v[i].imag();
	}
	return std::move(res);
}

template<typename, int> struct Geometry;

template<typename, int, typename > struct Field;

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

template<typename > struct is_complex
{
	static constexpr bool value = false;
};

template<typename T> struct is_complex<std::complex<T> >
{
	static constexpr bool value = true;
};

template<int TOP, typename TL, typename TR> struct is_complex<BiOp<TOP, TL, TR> >
{
	static constexpr bool value = is_complex<TL>::value || is_complex<TR>::value;
};

template<typename > struct is_real
{
	static constexpr bool value = false;
};

template<> struct is_real<Real>
{
	static constexpr bool value = true;
};

template<int TOP, typename TL, typename TR> struct is_real<BiOp<TOP, TL, TR> >
{
	static constexpr bool value = is_real<TL>::value && is_real<TR>::value;
};

template<typename > struct has_PlaceHolder
{
	static constexpr bool value = false;
};

template<typename TL>
struct is_arithmetic_scalar
{
	static constexpr bool value = (std::is_arithmetic<TL>::value || is_complex<TL>::value || has_PlaceHolder<TL>::value);
};

template<typename T>
struct is_primitive
{
	static constexpr bool value = is_arithmetic_scalar<T>::value;
};

template<int N, typename TE>
struct is_primitive<nTuple<N, TE> >
{
	static constexpr bool value = is_arithmetic_scalar<TE>::value;
};

//template<typename T>
//struct is_storage_type<std::complex<T>>
//{
//	static constexpr  bool value = false;
//};
//
//template<typename TG, typename T>
//struct is_storage_type<Field<TG, T> >
//{
//	static constexpr  bool value = is_storage_type<T>::value;
//};

template<typename T>
struct is_ntuple
{
	static constexpr bool value = false;
};

template<int N, typename T>
struct is_ntuple<nTuple<N, T>>
{
	static constexpr bool value = true;
};

template<typename T>
struct is_field_expression
{
	static constexpr bool value = false;
};

template<typename TG, int IF, int TOP, typename TL, typename TR>
struct is_field_expression<Field<TG, IF, BiOp<TOP, TL, TR> > >
{
	static constexpr bool value = true;
};

template<typename TG, int IF, int TOP, typename TL>
struct is_field_expression<Field<TG, IF, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

template<typename T>
struct is_ntuple_expression
{
	static constexpr bool value = false;
};

template<int N, int TOP, typename TL, typename TR>
struct is_ntuple_expression<nTuple<N, BiOp<TOP, TL, TR> > >
{
	static constexpr bool value = true;
};

template<int N, int TOP, typename TL>
struct is_field_expression<nTuple<N, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

template<typename T>
struct is_expression
{
	static constexpr bool value = is_ntuple_expression<T>::value || is_field_expression<T>::value;
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

typedef enum
{
	PLUS = 1, MINUS = 2, MULTIPLIES = 3, DIVIDES = 4, NEGATE = 5, RECIPROCAL,

	MODULUS, BITWISEXOR, BITWISEAND, BITWISEOR,

	// Tensor product
	TENSOR_PRODUCT, // outer product
	TENSOR_CONTRACTION,

	WEDGE = 20,
	HODGESTAR,
	EXTRIORDERIVATIVE,
	CODIFFERENTIAL,
	INTERIOR_PRODUCT,

	DOT,
	CROSS,
//
//
//	GRAD,
//	DIVERGE,
//	CURL,
//	CURLPDX,
//	CURLPDY,
//	CURLPDZ,

	MAPTO,

	EQUAL,
	LESS,
	GREATER,

	REAL,
	IMAGINE,

	NULL_OP

} OpType;

typedef enum
{
	SIN = NULL_OP + 1, COS, TAN, CTAN, EXP, LOG10, LOG2, LN, ABS

} MathFunType;

template<typename T>
inline auto Reciprocal(T const & f)
DECL_RET_TYPE(( 1.0/f))
namespace ops
{

struct plus
{
	template<typename TL, typename TR>
	constexpr auto operator()(TL const & l, TR const & r) const->decltype(l+r)
	{
		return l + r;
	}
};
struct minus
{
	template<typename TL, typename TR>
	constexpr auto operator()(TL const & l, TR const & r) const->decltype(l-r)
	{
		return l - r;
	}
};
struct multiplies
{
	template<typename TL, typename TR>
	constexpr auto operator()(TL const & l, TR const & r) const->decltype(l/r)
	{
		return l / r;
	}
};
struct divides
{
	template<typename TL, typename TR>
	constexpr auto operator()(TL const & l, TR const & r) const->decltype(l*r)
	{
		return l * r;
	}
};

struct negate
{
	template<typename TL>
	constexpr auto operator()(TL const & l) const->decltype(-l )
	{
		return -l;
	}
};

}  // namespace ops
//template<int TOP, typename TL, typename TR> struct OpTraits;
//
//template<typename T>
//struct FieldTraits
//{
//	enum
//	{
//		is_field = false
//	};
//
//	enum
//	{
//		IForm = 0
//	}
//	;
//	typedef T value_type;
//};
//
//template<typename TM, int IFORM, typename TExpr>
//struct FieldTraits<Field<TM, IFORM, TExpr> >
//{
//	typedef Field<TM, IFORM, TExpr> this_type;
//	enum
//	{
//		is_field = true
//	};
//
//	enum
//	{
//		IForm = IFORM
//	}
//	;
//	typedef typename this_type::value_type value_type;
//};

template<typename TL>
struct is_field
{
	static const bool value = false;
};

template<typename TG, int IF, typename TL>
struct is_field<Field<TG, IF, TL>>
{
	static const bool value = true;
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
