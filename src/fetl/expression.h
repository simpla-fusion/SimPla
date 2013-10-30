/*
 * expression.h
 *
 *  Created on: 2013-7-20
 *      Author: salmon
 */

#ifndef EXPRESSION_H_
#define EXPRESSION_H_

#include <fetl/field.h>
#include <complex>
#include <cstddef>
#include <type_traits>

/** !file
 *     Define type-independent expression template
 *     arithmetic operator + - / *
 *	   TODO: relation operator
 *
 *
 * */

namespace simpla
{

typedef enum
{
	PLUS = 1,

	MINUS = 2,

	MULTIPLIES = 3,

	DIVIDES = 4,

	WEDGE, CROSS, DOT,

	MODULUS, BITWISEXOR, BITWISEAND, BITWISEOR,

	GRAD, DIVERGE, CURL, HODGESTAR, EXTRIORDERIVATIVE, NEGATE, CURLPD1, CURLPD2,

	EQUAL, LESS, GREATER,

	NULL_OP

} OpType;

typedef enum
{
	SIN = NULL_OP + 1, COS, TAN, CTAN, EXP, LOG10, LOG2, LN, ABS

} MathFunType;

#define DECL_RET_TYPE(_EXPR_) ->decltype((_EXPR_)){return (_EXPR_);}

#define ENABLE_IF_DECL_RET_TYPE(_COND_,_EXPR_) \
        ->typename std::enable_if<_COND_,decltype((_EXPR_))>::type {return (_EXPR_);}

struct NullType;

struct EmptyType
{
};

struct Zero
{
};

struct One
{
};
struct Infinity
{
};

struct Undefine
{
};

template<typename TE> inline TE const &
operator +(TE const &e, Zero const &)
{
	return (e);
}

template<typename TE> inline TE const &
operator +(Zero const &, TE const &e)
{
	return (e);
}

template<typename TE> inline TE const &
operator -(TE const &e, Zero const &)
{
	return (e);
}

template<typename TE> inline auto operator -(Zero const &, TE const &e)
DECL_RET_TYPE (((-e)))

Zero operator +(Zero const &, Zero const &e)
{
	return (Zero());
}

template<typename TE> inline TE const &operator *(TE const &e, One const &)
{
	return (e);
}

template<typename TE> inline TE const & operator *(One const &, TE const &e)
{
	return (e);
}

template<typename TE> inline Zero operator *(TE const &, Zero const &)
{
	return (Zero());
}

template<typename TE> inline Zero operator *(Zero const &, TE const &)
{
	return (Zero());
}

template<typename TE> inline Infinity operator /(TE const &e, Zero const &)
{
	return (Infinity());
}

template<typename TE> inline Zero operator /(Zero const &, TE const &e)
{
	return (Zero());
}

template<typename TE> inline Zero operator /(TE const &, Infinity const &)
{
	return (Zero());
}

template<typename TE> inline Infinity operator /(Infinity const &, TE const &e)
{
	return (Infinity());
}

template<typename TL> inline auto   //
operator==(TL const & lhs, Zero)
DECL_RET_TYPE ((lhs))

template<typename TR> inline auto   //
operator==(Zero, TR const & rhs)
DECL_RET_TYPE ((rhs))

template<int N, typename TExpr> struct nTuple;

template<typename TG, typename TExpr> struct Field;

template<int TOP, typename TL, typename TR> struct BiOp;

template<int TOP, typename TL> struct UniOp;

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
	static const bool value = (std::is_arithmetic<TL>::value
			|| is_complex<TL>::value || has_PlaceHolder<TL>::value);
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
struct ReferenceTraits
{
	typedef typename remove_const_reference<T>::type TL;
	typedef typename std::conditional<is_storage_type<TL>::value, TL &, TL>::type type;
};

template<typename T>
struct ConstReferenceTraits
{
	typedef typename remove_const_reference<T>::type TL;
	typedef typename std::conditional<is_storage_type<TL>::value, TL const &,
			const TL>::type type;
};

template<>
struct ConstReferenceTraits<double>
{
	typedef double type;
};
inline double index(double v, size_t)
{
	return (v);
}
inline std::complex<double> index(std::complex<double> v, size_t)
{
	return (v);
}
template<typename T> inline
auto index(T const & v, size_t s)->decltype(v[s])
{
	return (v[s]);
}

template<bool cond>
struct c_index
{
	template<typename T>
	static auto eval(T const & v, size_t s)
	DECL_RET_TYPE (v[s])
};

template<>
struct c_index<false>
{
	template<typename T> static auto eval(T const & v, size_t)
	DECL_RET_TYPE(v)
};

//template<typename TOP, typename TL>
//struct UniOp
//{
//public:
//
//	typename ConstReferenceTraits<TL>::type expr;
//
//	typedef UniOp<TOP, TL> ThisType;
//
//	typedef decltype(TOP::eval(expr ,0 )) ValueType;
//
//	UniOp(TL const & l) :
//			expr(l)
//	{
//	}
//
//	UniOp(ThisType const &) =default;
//
//	inline ValueType
//	operator[](size_t s)const
//	{
//		return ( TOP::eval(expr,s ));
//	}
//
//};
//
//struct OpNegate
//{
//	template<typename TL> inline static auto eval(TL const & l, size_t s)
//	DECL_RET_TYPE((-index(l, s)))
//};
//
//#define UNI_FUN_OP(_OP_NAME_,_FUN_)                                                    \
//struct Op##_OP_NAME_                                                                 \
//{template<typename TL> inline static auto                               \
//eval(TL const & l, size_t s) ->decltype((_OP_NAME_(index(l, s))))     \
//{	return (_FUN_(index(l, s)));} };                                                     \
//template<typename TL> inline    \
//typename std::enable_if<!is_arithmetic_scalar<TL>::value,UniOp<Op##_OP_NAME_, TL> >::type \
//_OP_NAME_(TL const &lhs){	return (UniOp<Op##_OP_NAME_, TL>(lhs));}                                           \
//
//
//UNI_FUN_OP(abs, std::abs)
//UNI_FUN_OP(sin, std::sin)
//UNI_FUN_OP(cos, std::cos)
//UNI_FUN_OP(tan, std::tan)
//UNI_FUN_OP(log, std::log)
//UNI_FUN_OP(log10, std::log10)
//UNI_FUN_OP(exp, std::exp)
//
//#undef UNI_FUN_OP
//
//inline double abs(std::complex<double> lhs)
//{
//	return (std::abs(lhs));
//}
//
//#define BI_FUN_OP(_OP_NAME_,_FUN_)                                               \
//struct Op##_OP_NAME_                                                            \
//{template<typename TL, typename TR> inline static auto eval(TL const &l,        \
//		TR const & r,size_t s) DECL_RET_TYPE((_FUN_(index(l,s),index(r,s))))};     \
//template<typename TL, typename TR> inline \
//typename std::enable_if<!(is_arithmetic_scalar<TL>::value && is_arithmetic_scalar<TR>::value), \
//	BiOp<Op##_OP_NAME_, TL, TR> >::type           \
//_OP_NAME_(TL const &lhs, TR const & rhs)                                         \
//{                                                                                \
//	return (BiOp<Op##_OP_NAME_, TL, TR>(lhs, rhs));                              \
//}                                                                                \
//
//BI_FUN_OP(pow, std::pow)
//BI_FUN_OP(modf, std::modf)
//BI_FUN_OP(Dot, Dot)
//BI_FUN_OP(Cross, Cross)
//#undef BI_FUN_OP

}
// namespace simpla

#endif /* EXPRESSION_H_ */
