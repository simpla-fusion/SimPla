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

template<typename TL>
struct is_arithmetic_scalar
{
	static const bool value = (std::is_arithmetic<TL>::value
			|| std::is_arithmetic<TL>::value);
};

template<typename TL>
struct ReferenceTraits
{
	typedef typename std::conditional<
			std::is_copy_constructible<TL>::value
					&& !(std::is_trivial<TL>::value
							&& sizeof(TL) > sizeof(int) * 3), TL, TL const &>::type type;

};

#define DECL_RET_TYPE(_EXPR_) ->decltype((_EXPR_)){return (_EXPR_);}

template<typename T, typename IDX> inline
auto index(const T& f, IDX const & idx) DECL_RET_TYPE((f[idx]))

template<typename T, typename IDX> inline T index(const T f[], IDX const & idx)
{
	return (f[idx]);
}

template<typename IDX> inline double index(double v, IDX)
{
	return (v);
}

template<typename IDX> inline std::complex<double> index(std::complex<double> v,
		IDX)
{
	return (v);
}

template<typename TOP, typename TL, typename TR> class BiOp
{
public:

	typename ReferenceTraits<TL>::type l_;
	typename ReferenceTraits<TR>::type r_;

	typedef BiOp<TOP, TL, TR> ThisType;

	typedef decltype(TOP::eval(index(l_,0),index(r_,0))) Value;

	BiOp(TL const & l, TR const &r) :
			l_(l), r_(r)
	{
	}

	BiOp(ThisType const &) =default;

	template<typename IDX> inline
	Value operator[](IDX const & idx)const
	{
		return (TOP::eval(index(l_,idx),index(r_,idx)));
	}

};
struct OpMultiplication
{
	template<typename TL, typename TR> inline static auto eval(TL const & l,
			TR const &r)
			DECL_RET_TYPE(l*r)
};
struct OpDivision
{
	template<typename TL, typename TR> inline static auto eval(TL const & l,
			TR const &r)
			DECL_RET_TYPE(l/r)
};

struct OpAddition
{
	template<typename TL, typename TR> inline static auto eval(TL const & l,
			TR const &r)
			DECL_RET_TYPE(l+r)
};
struct OpSubtraction
{
	template<typename TL, typename TR> inline static auto eval(TL const & l,
			TR const &r)
			DECL_RET_TYPE(l-r)
};

#define EXCLUDE_ARITHEMTIC( _TYPE_ )

template<typename TL, typename TR> inline typename std::enable_if<
		!(is_arithmetic_scalar<TL>::value && is_arithmetic_scalar<TR>::value),
		BiOp<OpAddition, TL, TR> >::type //
operator +(TL const &lhs, TR const & rhs)
{
	return (BiOp<OpAddition, TL, TR>(lhs, rhs));
}
template<typename TL, typename TR> inline typename std::enable_if<
		!(is_arithmetic_scalar<TL>::value && is_arithmetic_scalar<TR>::value),
		BiOp<OpSubtraction, TL, TR> >::type //
operator -(TL const &lhs, TR const & rhs)
{
	return (BiOp<OpSubtraction, TL, TR>(lhs, rhs));
}
template<typename TL, typename TR> inline typename std::enable_if<
		!(is_arithmetic_scalar<TL>::value && is_arithmetic_scalar<TR>::value),
		BiOp<OpMultiplication, TL, TR> >::type //
operator *(TL const &lhs, TR const & rhs)
{
	return (BiOp<OpMultiplication, TL, TR>(lhs, rhs));
}
template<typename TL, typename TR> inline typename std::enable_if<
		!(is_arithmetic_scalar<TL>::value && is_arithmetic_scalar<TR>::value),
		BiOp<OpDivision, TL, TR> >::type //
operator /(TL const &lhs, TR const & rhs)
{
	return (BiOp<OpDivision, TL, TR>(lhs, rhs));
}

template<typename TL> inline auto //
operator -(TL const &lhs) DECL_RET_TYPE((-1.0*lhs))

#define BI_FUN_OP(_OP_NAME_,_FUN_)                                                       \
                                                                                     \
struct Op##_OP_NAME_                                                                 \
{                                                                                    \
	template<typename TL, typename TR> inline static auto eval(TL const & l,         \
			TR const &r)->decltype(_FUN_(l, r))                                        \
	{                                                                                \
		return (_FUN_(l, r));                                               \
	}                                                                                \
};                                                                                   \
template<typename TL, typename TR> inline typename std::enable_if<                   \
		!(is_arithmetic_scalar<TL>::value && is_arithmetic_scalar<TR>::value),       \
		BiOp<Op##_OP_NAME_, TL, TR> >::type                                        \
_OP_NAME_(TL const &lhs, TR const & rhs)                                                   \
{                                                                                    \
	return (BiOp<Op##_OP_NAME_, TL, TR>(lhs, rhs));                                  \
}                                                                                    \

BI_FUN_OP(pow, std::pow)
BI_FUN_OP(modf, std::modf)
BI_FUN_OP(Dot, Dot)
BI_FUN_OP(Cross, Cross)

#undef BI_FUN_OP

template<typename TOP, typename TL> class UniOp
{
public:
	typename ReferenceTraits<TL>::type l_;

	typedef UniOp<TOP, TL> ThisType;

	typedef decltype(TOP::eval(index(l_,0))) Value;

	UniOp(TL const & l) :
			l_(l)
	{
	}

	UniOp(ThisType const &) =default;

	template<typename IDX> inline Value operator[](IDX const & idx) const
	{
		return (TOP::eval(index(l_, idx)));
	}

};

#define UNI_FUN_OP(_OP_NAME_,_FUN_)                                                       \
struct Op##_OP_NAME_                                                                \
{                                                                                   \
	template<typename TL> inline static auto eval(TL const & l)                     \
	->decltype((_FUN_(l)))                                                 \
	{ return (_FUN_(l));  }                                                \
};                                                                                  \
                                                                                    \
template<typename TL> inline                                           \
typename std::enable_if< !(is_arithmetic_scalar<TL>::value),UniOp<Op##_OP_NAME_, TL> >::type   \
_OP_NAME_(TL const &lhs)                                                                  \
{                                                                                   \
	return (UniOp<Op##_OP_NAME_, TL>(lhs));                                         \
}
UNI_FUN_OP(abs, std::abs)
UNI_FUN_OP(sin, std::sin)
UNI_FUN_OP(cos, std::cos)
UNI_FUN_OP(tan, std::tan)
UNI_FUN_OP(log, std::log)
UNI_FUN_OP(log10, std::log10)
UNI_FUN_OP(exp, std::exp)

#undef UNI_FUN_OP

}
// namespace simpla

#endif /* EXPRESSION_H_ */
