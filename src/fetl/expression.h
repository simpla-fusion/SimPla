/*
 * expression.h
 *
 *  Created on: 2013-7-20
 *      Author: salmon
 */

#ifndef EXPRESSION_H_
#define EXPRESSION_H_

#include <type_traits>
#include <utility>
#include <complex>
#include <cmath>
/** !file
 *     Define type-independent expression template
 *     arithmetic operator + - / *
 *	   TODO: relation operator
 *
 *
 * */

namespace simpla
{

struct NullType;

struct EmptyType
{
};

struct Zero
{
	Zero()
	{
	}
	template<typename TL> Zero(TL const &)
	{
	}
	template<typename TL, typename TR> Zero(TL const &, TR const &)
	{
	}
};
struct One
{
	One()
	{
	}
	template<typename TL> One(TL const &)
	{
	}

	template<typename TL, typename TR> One(TL const &, TR const &)
	{
	}

};

struct Infinity
{
	Infinity()
	{

	}
	template<typename TL> Infinity(TL const &)
	{
	}
	template<typename TL, typename TR> Infinity(TL const &, TR const &)
	{
	}

};

struct Undefine
{
	Undefine()
	{

	}
	template<typename TL> Undefine(TL const &)
	{
	}
	template<typename TL, typename TR> Undefine(TL const &, TR const &)
	{
	}

};

template<int N, typename TExpr> struct nTuple;

template<typename TG, typename TExpr> struct Field;

template<typename > struct PlaceHolder;

template<typename TOP, typename TL, typename TR> struct BiOp;

template<typename TOP, typename TL> struct UniOp;

template<typename T>
struct remove_const_reference
{
	typedef typename std::remove_const<typename std::remove_reference<T>::type>::type type;
};

// check is_Field
template<typename > struct is_Field
{
	static const bool value = false;
};
template<typename T> struct is_Field<T&>
{
	static const bool value = is_Field<T>::value;
};

template<typename T> struct is_Field<const T>
{
	static const bool value = is_Field<T>::value;
};

template<typename TG, typename TE> struct is_Field<Field<TG, TE> >
{
	static const bool value = true;
};

template<typename TOP, typename TL, typename TR> struct is_Field<
		BiOp<TOP, TL, TR> >
{
	static const bool value = is_Field<TL>::value || is_Field<TR>::value;
};

// check is_nTuple
template<typename > struct is_nTuple
{
	static const bool value = false;
};

template<typename T> struct is_nTuple<T&>
{
	static const bool value = is_nTuple<T>::value;
};

template<typename T> struct is_nTuple<const T>
{
	static const bool value = is_nTuple<T>::value;
};

template<int N, typename T> struct is_nTuple<nTuple<N, T> >
{
	static const bool value = true;
};

template<typename TOP, typename TL, typename TR> struct is_nTuple<
		BiOp<TOP, TL, TR> >
{
	static const bool value = (is_nTuple<TL>::value || is_nTuple<TR>::value)
			&& !(is_Field<TL>::value || is_Field<TR>::value);
};

template<typename TOP, typename TExpr> struct is_nTuple<UniOp<TOP, TExpr> >
{
	static const bool value = is_nTuple<TExpr>::value;
};

template<typename > struct is_complex
{
	static const bool value = false;
};

template<typename T> struct is_complex<std::complex<T> >
{
	static const bool value = true;
};

template<typename TOP, typename TL, typename TR> struct is_complex<
		BiOp<TOP, TL, TR> >
{
	static const bool value = is_complex<TL>::value || is_complex<TR>::value;
};

template<typename TOP, typename TExpr> struct is_complex<UniOp<TOP, TExpr> >
{
	static const bool value = is_complex<TExpr>::value;
};

template<typename > struct has_PlaceHolder
{
	static const bool value = false;
};

template<typename TE> struct has_PlaceHolder<PlaceHolder<TE> >
{
	static const bool value = true;
};

template<typename TOP, typename TL, typename TR> struct has_PlaceHolder<
		BiOp<TOP, TL, TR> >
{
	static const bool value = has_PlaceHolder<TL>::value
			|| has_PlaceHolder<TR>::value;
};

template<typename TOP, typename TExpr> struct has_PlaceHolder<UniOp<TOP, TExpr> >
{
	static const bool value = has_PlaceHolder<TExpr>::value;
};

template<typename TE> struct is_indexable
{
	typedef typename remove_const_reference<TE>::type T;
	static const bool value = is_nTuple<T>::value || is_Field<T>::value
			|| std::is_pointer<T>::value;
};

template<typename TL>
struct is_arithmetic_scalar
{
	static const bool value = (std::is_arithmetic<TL>::value
			|| is_complex<TL>::value || has_PlaceHolder<TL>::value);
};

template<typename T>
struct is_storage_type
{
	static const bool value = false;
};

template<typename TL>
struct is_primitive
{
	static const bool value = (is_arithmetic_scalar<TL>::value
			|| is_nTuple<TL>::value || has_PlaceHolder<TL>::value);
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

#define DECL_RET_TYPE(_EXPR_) ->decltype((_EXPR_)){return (_EXPR_);}
#define ENABLE_IF_DECL_RET_TYPE(_COND_,_EXPR_) \
        ->typename std::enable_if<_COND_,decltype((_EXPR_))>::type {return (_EXPR_);}

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
