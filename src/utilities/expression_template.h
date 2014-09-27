/*
 * expression_template.h
 *
 *  Created on: 2014年9月26日
 *      Author: salmon
 */

#ifndef EXPRESSION_TEMPLATE_H_
#define EXPRESSION_TEMPLATE_H_

#include <cmath>
#include <limits>
#include <type_traits>
#include <complex>
#include "sp_type_traits.h"

namespace simpla
{
template<typename ... > class Expression;

template<typename T>
struct is_expresson
{
	HAS_MEMBER(is_expression);
	static constexpr bool value=has_member_is_expression<T>::value;
};

namespace _impl
{

template<typename T>
struct reference_traits
{
	static constexpr bool no_refercne = std::is_pointer<T>::value

	|| std::is_reference<T>::value

	|| std::is_scalar<T>::value

	|| is_expresson<T>::value;

	typedef typename std::conditional<no_refercne, T, T&>::type type;

	typedef typename std::conditional<no_refercne, T, const T&>::type const_reference;

	typedef typename std::conditional<no_refercne, T, T&>::type reference;

};

}  // namespace _impl

template<typename TOP, typename TL, typename TR>
struct Expression<TOP, TL, TR>
{
	typename _impl::reference_traits<TL>::const_reference lhs;
	typename _impl::reference_traits<TR>::const_reference rhs;
	TOP op_;

	Expression(TL const & l, TR const & r) :
			lhs(l), rhs(r), op_()
	{
	}

	Expression(TOP op, TL const & l, TR const & r) :
			lhs(l), rhs(r), op_(op)
	{
	}
	~Expression()
	{
	}

	template<typename IndexType>
	inline auto operator[](IndexType const &s) const
	DECL_RET_TYPE ((op_(get_value(lhs, s), get_value(rhs, s))))
}
;

///   \brief  Unary operation
template<typename TOP, typename TL>
struct Expression<TOP, TL>
{

	typename _impl::reference_traits<TL>::const_reference lhs;

	TOP op_;

	Expression(TL const & l) :
			lhs(l), op_()
	{
	}
	Expression(TOP op, TL const & l) :
			lhs(l), op_(op)
	{
	}
	~Expression()
	{
	}

	template<typename IndexType>
	inline auto operator[](IndexType const &s) const
	DECL_RET_TYPE ((op_(get_value(lhs, s))))

};

template<typename ... T, typename TI, template<typename > class F>
auto get_value(F<Expression<T...>> & expr, TI const & s)
->typename std::remove_reference<decltype(expr[s])>::type
{
	return expr[s];
}
namespace _impl
{
struct binary_right
{
	template<typename TL, typename TR>
	TR const &operator()(TL const &, TR const & r) const
	{
		return r;
	}
};

struct _swap
{
	template<typename TL, typename TR>
	void operator()(TL & l, TR & r) const
	{
		std::swap(l, r);
	}
};

struct _assign
{
	template<typename TL, typename TR>
	void operator()(TL & l, TR const& r) const
	{
		l = r;
	}
};

#define DEF_BOP(_NAME_,_OP_)                                                               \
struct _NAME_                                                                             \
{                                                                                              \
	template<typename TL, typename TR>                                                         \
	constexpr auto operator()(TL const& l, TR const & r) const->decltype(l _OP_ r)                 \
	{  return l _OP_ r;   }                                                                       \
};                                                                                             \


#define DEF_UOP(_NAME_,_OP_)     \
struct _NAME_                                                                             \
{                                                                                              \
	template<typename TL >                                                         \
	constexpr auto operator()(TL const & l ) const->decltype(_OP_ l )                 \
	{  return  _OP_ l;   }                                                                                          \
};

DEF_BOP(plus, +)
DEF_BOP(minus, -)
DEF_BOP(multiplies, *)
DEF_BOP(divides, /)
DEF_BOP(modulus, %)

DEF_UOP(negate, -)
DEF_UOP(unary_plus, +)

DEF_BOP(bitwise_and, &)
DEF_BOP(bitwise_or, |)
DEF_BOP(bitwise_xor, ^)
DEF_UOP(bitwise_not, ~)

DEF_BOP(shift_left, <<)
DEF_BOP(shift_right, >>)

DEF_UOP(logical_not, !)
DEF_BOP(logical_and, &&)
DEF_BOP(logical_or, ||)
DEF_BOP(not_equal_to, !=)
DEF_BOP(greater, >)
DEF_BOP(less, <)
DEF_BOP(greater_equal, >=)
DEF_BOP(less_equal, <=)

#undef DEF_UOP
#undef DEF_BOP

#define DEF_BOP(_NAME_,_OP_)                                                               \
struct _NAME_                                                                             \
{                                                                                              \
	template<typename TL, typename TR>                                                         \
	constexpr auto operator()(TL  & l, TR const & r) const->decltype(l _OP_ r)                 \
	{  return l _OP_ r;   }                                                                       \
};

DEF_BOP(plus_assign, +=)
DEF_BOP(minus_assign, -=)
DEF_BOP(multiplies_assign, *=)
DEF_BOP(divides_assign, /=)
DEF_BOP(modulus_assign, %=)

#undef DEF_BOP
struct equal_to
{
	template<typename TL, typename TR>
	constexpr bool operator()(TL const & l, TR const & r) const
	{
		return l == r;
	}
	constexpr bool operator()(double l, double r) const
	{
		return std::abs(l - r) <= std::numeric_limits<double>::epsilon();
	}
};

// The few binary functions we miss.
struct _atan2
{
	template<typename _Tp>
	_Tp operator()(const _Tp& x, const _Tp& y) const
	{
		return std::atan2(x, y);
	}
};

struct _pow
{
	template<typename _Tp>
	_Tp operator()(const _Tp& x, const _Tp& y) const
	{
		return std::pow(x, y);
	}
};

// Implementations of unary functions applied to valarray<>s.

struct _abs
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::abs(__t);
	}
};

struct _cos
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::cos(__t);
	}
};

struct _acos
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::acos(__t);
	}
};

struct _cosh
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::cosh(__t);
	}
};

struct _sin
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::sin(__t);
	}
};

struct _asin
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::asin(__t);
	}
};

struct _sinh
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::sinh(__t);
	}
};

struct _tan
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::tan(__t);
	}
};

struct _atan
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::atan(__t);
	}
};

struct _tanh
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::tanh(__t);
	}
};

struct _exp
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::exp(__t);
	}
};

struct _log
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::log(__t);
	}
};

struct _log10
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::log10(__t);
	}
};

struct _sqrt
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::sqrt(__t);
	}
};

struct _real
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::real(__t);
	}
};
struct _imag
{
	template<typename _Tp>
	_Tp operator()(const _Tp& __t) const
	{
		return std::imag(__t);
	}
};

class _inner_product
{
};
}  // namespace _impl

#define _SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(_OP_,_OBJ_,_NAME_)                                                  \
	template<typename ...T1,typename  T2> auto operator _OP_(_OBJ_<T1...> const & l,T2 const &r)  \
	DECL_RET_TYPE((_OBJ_<Expression<_impl::_NAME_,_OBJ_<T1...>,T2>>(l,r)))                  \


#define _SP_DEFINE_EXPR_BINARY_OPERATOR(_OP_,_OBJ_,_NAME_)                                                  \
	template<typename ...T1,typename  T2> auto operator _OP_(_OBJ_<T1...> const & l,T2 const &r)  \
	DECL_RET_TYPE((_OBJ_<Expression<_impl::_NAME_,_OBJ_<T1...>,T2>>(l,r)))                  \
	template< typename T1,typename ...T2> auto operator _OP_(T1 const & l, _OBJ_< T2...>const &r)                    \
	DECL_RET_TYPE((_OBJ_<Expression< _impl::_NAME_,T1,_OBJ_< T2...>>>(l,r)))                  \
	template< typename ... T1,typename ...T2> auto operator _OP_(_OBJ_< T1...> const & l,_OBJ_< T2...>  const &r)                    \
	DECL_RET_TYPE((_OBJ_<Expression< _impl::_NAME_,_OBJ_< T1...>,_OBJ_< T2...>>>(l,r)))                  \


#define _SP_DEFINE_EXPR_UNARY_OPERATOR(_OP_,_OBJ_,_NAME_)                           \
		template<typename ...T> auto operator _OP_(_OBJ_<T...> const &l)  \
		DECL_RET_TYPE((_OBJ_<Expression<_impl::_NAME_,_OBJ_<T...> >>(l)))   \

#define _SP_DEFINE_EXPR_BINARY_FUNCTION(_NAME_,_OBJ_)                                                  \
			template<typename ...T1,typename  T2> auto   _NAME_(_OBJ_<T1...> const & l,T2 const &r)  \
			DECL_RET_TYPE((_OBJ_<Expression<_impl::_##_NAME_,_OBJ_<T1...>,T2>>(l,r)))                  \
			template< typename T1,typename ...T2> auto _NAME_(T1 const & l, _OBJ_< T2...>const &r)                    \
			DECL_RET_TYPE((_OBJ_<Expression< _impl::_##_NAME_,T1,_OBJ_< T2...>>>(l,r)))                  \
			template< typename ... T1,typename ...T2> auto _NAME_(_OBJ_< T1...> const & l,_OBJ_< T2...>  const &r)                    \
			DECL_RET_TYPE((_OBJ_<Expression< _impl::_##_NAME_,_OBJ_< T1...>,_OBJ_< T2...>>>(l,r)))                  \


#define _SP_DEFINE_EXPR_UNARY_FUNCTION( _NAME_,_OBJ_)                           \
		template<typename ...T> auto   _NAME_(_OBJ_<T ...> const &r)  \
		DECL_RET_TYPE((_OBJ_<Expression<_impl::_##_NAME_,_OBJ_<T ...>>>(r)))   \


#define  DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_CONCEPT_)                                              \
_SP_DEFINE_EXPR_BINARY_OPERATOR(+, _CONCEPT_, plus)                                      \
_SP_DEFINE_EXPR_BINARY_OPERATOR(-, _CONCEPT_, minus)                                     \
_SP_DEFINE_EXPR_BINARY_OPERATOR(*, _CONCEPT_, multiplies)                                \
_SP_DEFINE_EXPR_BINARY_OPERATOR(/, _CONCEPT_, divides)                                   \
_SP_DEFINE_EXPR_BINARY_OPERATOR(%, _CONCEPT_, modulus)                                   \
_SP_DEFINE_EXPR_BINARY_OPERATOR(^, _CONCEPT_, bitwise_xor)                               \
_SP_DEFINE_EXPR_BINARY_OPERATOR(&, _CONCEPT_, bitwise_and)                               \
_SP_DEFINE_EXPR_BINARY_OPERATOR(|, _CONCEPT_, bitwise_or)                                \
_SP_DEFINE_EXPR_UNARY_OPERATOR(~, _CONCEPT_, bitwise_not)                                \
_SP_DEFINE_EXPR_BINARY_OPERATOR(&&, _CONCEPT_, logical_and)                              \
_SP_DEFINE_EXPR_BINARY_OPERATOR(||, _CONCEPT_, logical_or)                               \
_SP_DEFINE_EXPR_UNARY_OPERATOR(+, _CONCEPT_, unary_plus)                                 \
_SP_DEFINE_EXPR_UNARY_OPERATOR(-, _CONCEPT_, negate)                                     \
_SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(<<, _CONCEPT_, shift_left)                               \
_SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(>>, _CONCEPT_, shift_right)                              \
_SP_DEFINE_EXPR_BINARY_FUNCTION(atan2, _CONCEPT_)        								 \
_SP_DEFINE_EXPR_BINARY_FUNCTION(pow, _CONCEPT_)          								 \
_SP_DEFINE_EXPR_UNARY_FUNCTION(abs, _CONCEPT_)                                           \
_SP_DEFINE_EXPR_UNARY_FUNCTION(cos, _CONCEPT_)                                           \
_SP_DEFINE_EXPR_UNARY_FUNCTION(acos, _CONCEPT_)                                          \
_SP_DEFINE_EXPR_UNARY_FUNCTION(cosh, _CONCEPT_)                                          \
_SP_DEFINE_EXPR_UNARY_FUNCTION(sin, _CONCEPT_)                                           \
_SP_DEFINE_EXPR_UNARY_FUNCTION(asin, _CONCEPT_)                                          \
_SP_DEFINE_EXPR_UNARY_FUNCTION(sinh, _CONCEPT_)                                          \
_SP_DEFINE_EXPR_UNARY_FUNCTION(tan, _CONCEPT_)                                           \
_SP_DEFINE_EXPR_UNARY_FUNCTION(tanh, _CONCEPT_)                                          \
_SP_DEFINE_EXPR_UNARY_FUNCTION(atan, _CONCEPT_)                                          \
_SP_DEFINE_EXPR_UNARY_FUNCTION(exp, _CONCEPT_)                                           \
_SP_DEFINE_EXPR_UNARY_FUNCTION(log, _CONCEPT_)                                           \
_SP_DEFINE_EXPR_UNARY_FUNCTION(log10, _CONCEPT_)                                         \
_SP_DEFINE_EXPR_UNARY_FUNCTION(sqrt, _CONCEPT_)                                          \
_SP_DEFINE_EXPR_UNARY_FUNCTION(real, _CONCEPT_)                                          \
_SP_DEFINE_EXPR_UNARY_FUNCTION(imag, _CONCEPT_)                                          \

//_SP_DEFINE_EXPR_BINARY_OPERATOR(!=, _CONCEPT_, not_equal_to)                             \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(==, _CONCEPT_, equal_to)                                 \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(<, _CONCEPT_, less)                                      \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(>, _CONCEPT_, greater)                                   \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(<=, _CONCEPT_, less_equal)                               \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(>=, _CONCEPT_, greater_equal)

//#undef _SP_DEFINE_EXPR_BINARY_OPERATOR
//#undef _SP_DEFINE_EXPR_UNARY_OPERATOR
//#undef _SP_DEFINE_EXPR_UNARY_FUNCTION

}
// namespace simpla

#endif /* EXPRESSION_TEMPLATE_H_ */
