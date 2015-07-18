/**
 * @file expression_template.h
 *
 *  Created on: 2014-9-26
 *      Author: salmon
 */

#ifndef EXPRESSION_TEMPLATE_H_
#define EXPRESSION_TEMPLATE_H_

#include <cmath>
#include <limits>
#include <type_traits>
#include <complex>
#include <tuple>
#include "type_traits.h"

namespace simpla
{

/**
 *  @ingroup gtl
 *  @addtogroup expression_template  Expression Template
 *  @{
 */
template<typename ...>class Expression;

template<typename ...> struct BooleanExpression;
template<typename ...> struct AssignmentExpression;

template<typename TOP, typename ...Args>
struct Expression<TOP, Args...>
{
	typedef Expression<TOP, Args...> this_type;

	typename std::tuple<traits::reference_t<Args> ...> args;

	TOP m_op_;

	Expression(this_type const & that) :
			args(that.args), m_op_(that.m_op_)
	{
	}
	Expression(this_type && that) :
			args(that.args), m_op_(that.m_op_)
	{
	}
	Expression(Args const&... pargs) :
			args(pargs ...), m_op_()
	{
	}

	Expression(TOP op, Args &&... pargs) :
			args(pargs ...), m_op_(op)
	{
	}

	~Expression()
	{
	}

}
;
template<typename TOP, typename TL, typename TR>
class BooleanExpression<TOP, TL, TR> : public Expression<TOP, TL, TR>
{
	using Expression<TOP, TL, TR>::Expression;

	operator bool() const
	{
		return false;
	}
};

template<typename TOP, typename TL>
class BooleanExpression<TOP, TL> : public Expression<TOP, TL>
{
	using Expression<TOP, TL>::Expression;

	operator bool() const
	{
		return false;
	}
};

template<typename TOP, typename TL, typename TR>
struct AssignmentExpression<TOP, TL, TR>
{
	typedef AssignmentExpression<TOP, TL, TR> this_type;
	TL & lhs;
	typename traits::reference<TR>::type rhs;
	TOP op_;

	AssignmentExpression(this_type const & that) :
			lhs(that.lhs), rhs(that.rhs), op_(that.op_)
	{
	}
	AssignmentExpression(this_type && that) :
			lhs(that.lhs), rhs(that.rhs), op_(that.op_)
	{
	}

	AssignmentExpression(TL & l, TR const& r) :
			lhs(l), rhs(r), op_()
	{
	}
	AssignmentExpression(TOP op, TL & l, TR const& r) :
			lhs(l), rhs(r), op_(op)
	{
	}

	~AssignmentExpression()
	{
	}

	template<typename IndexType>
	inline auto operator[](IndexType const &s) const
	DECL_RET_TYPE ((op_(traits::index(lhs, s), traits::index(rhs, s))))
//			DECL_RET_TYPE ((op_( lhs, rhs, s )))

};

namespace traits
{

template<typename T>
struct is_expresson
{
	static constexpr bool value = false;
};
template<typename ...T, template<typename ...> class F>
struct is_expresson< F <Expression<T... > > >
{
	static constexpr bool value=true;
};

template<typename ...T>
struct is_expresson<Expression<T...> >
{
	static constexpr bool value = true;
};
}
// namespace traits
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
struct null_op
{
	template<typename TL>
	TL const &operator()(TL const & l) const
	{
		return l;
	}
};

struct _swap
{
	template<typename TL, typename TR>
	void operator()(TL & l, TR & r) const
	{
		std::swap(l, r);
	}

	template<typename TL, typename TR, typename TI>
	void operator()(TL & l, TR & r, TI const &s) const
	{
		std::swap(traits::index(l, s), traits::index(r, s));
	}
};

#define DEF_BOP(_NAME_,_OP_)                                                               \
struct _NAME_                                                                             \
{                                                                                              \
	template<typename TL, typename TR>                                                         \
	constexpr auto operator()(TL const& l, TR const & r) const->decltype(l _OP_ r)                 \
	{  return l _OP_ r;   }                                                                       \
	template<typename TL, typename TR,typename TI>                                                         \
	constexpr auto operator()(TL const& l, TR const & r,TI const & s) const         \
	->decltype(traits::index(l,s) _OP_ traits::index( r,s) )                 \
	{  return traits::index(l,s) _OP_ traits::index( r,s);   }                                    \
};

#define DEF_UOP(_NAME_,_OP_)                         														\
struct _NAME_                                                                             \
{                                                                                              \
	template<typename TL >                                                         \
	constexpr auto operator()(TL const & l ) const->decltype(_OP_ l )                 \
	{  return  _OP_ l;   }                                                   \
	template<typename TL,typename TI >                                                         \
	constexpr auto operator()(TL const & l, TI const & s) const->decltype(_OP_ traits::index( l ,s)) \
	{  return  _OP_  traits::index( l ,s);   } \
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
/**
 * ### Assignment Operator
 *
 *   Pseudo-Signature 	 				         | Semantics
 *  ---------------------------------------------|--------------
 *  `operator+=(Object &,Expression const &)`     | Assign operation +
 *  `operator-=(Object & ,Expression const &)`     | Assign operation -
 *  `operator/=(Object & ,Expression const &)`     | Assign operation /
 *  `operator*=(Object & ,Expression const &)`     | Assign operation *
 */
#define DEF_ASSIGN_OP(_NAME_,_OP_)                                                               \
struct _NAME_                                                                             \
{                                                                                              \
	template<typename TL, typename TR>                                                         \
	  void operator()(TL  & l, TR const & r) const                  \
	{ l _OP_ r ; }           \
	template<typename TL, typename TR,typename TI>                                                         \
	  void operator()(TL  & l, TR const & r,TI const & s)const           \
	{    traits::index(l,s) _OP_ traits::index( r,s)   ;    }    \
};

//DEF_ASSIGN_OP(_assign, =)
DEF_ASSIGN_OP(plus_assign, +=)
DEF_ASSIGN_OP(minus_assign, -=)
DEF_ASSIGN_OP(multiplies_assign, *=)
DEF_ASSIGN_OP(divides_assign, /=)
DEF_ASSIGN_OP(modulus_assign, %=)

#undef DEF_ASSIGN_OP

struct _assign
{
	template<typename TL, typename TR>
	void operator()(TL & l, TR const & r) const
	{
		l = r;
	}
	template<typename TL, typename TR, typename TI>
	void operator()(TL & l, TR const & r, TI const & s) const
	{
		traits::index(l, s) = traits::index(r, s);
	}
};
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
template<typename TOP> struct op_traits
{
	typedef logical_and type;
};
template<> struct op_traits<not_equal_to>
{
	typedef logical_or type;
};

#define DEF_STD_BINARY_FUNCTION(_NAME_ )                                                               \
struct _##_NAME_                                                                             \
{                                                                                              \
	template<typename TL, typename TR>                                                         \
	constexpr auto operator()(TL const& l, TR const & r) const->decltype(_NAME_(l,  r))                 \
	{  return std::_NAME_(l,  r);   }                                                                       \
	template<typename TL, typename TR,typename TI>                                                         \
	constexpr auto operator()(TL const& l, TR const & r,TI const & s) const         \
	->decltype(std::_NAME_(traits::index(l,s) , traits::index( r,s) ))                 \
	{  return std::_NAME_(traits::index(l,s) , traits::index( r,s) );   }                                    \
};

DEF_STD_BINARY_FUNCTION(atan2)
DEF_STD_BINARY_FUNCTION(pow)

#undef DEF_STD_BINARY_FUNCTION

#define DEF_UNARY_FUNCTION(_NAME_ )                                                               \
struct _##_NAME_                                                                             \
{                                                                                              \
	template<typename TL >                                                         \
	constexpr auto operator()(TL const& l ) const->decltype(_NAME_(l ))                 \
	{  return std::_NAME_(l );   }                                                                       \
	template<typename TL ,typename TI>                                                         \
	constexpr auto operator()(TL const& l, TI const & s) const         \
	->decltype(std::_NAME_(traits::index(l,s)   ))                 \
	{  return std::_NAME_(traits::index(l,s)  );   }                                    \
};

//DEF_UNARY_FUNCTION(fabs)
//DEF_UNARY_FUNCTION(abs)
DEF_UNARY_FUNCTION(cos)
DEF_UNARY_FUNCTION(acos)
DEF_UNARY_FUNCTION(cosh)
DEF_UNARY_FUNCTION(sin)
DEF_UNARY_FUNCTION(asin)
DEF_UNARY_FUNCTION(sinh)
DEF_UNARY_FUNCTION(tan)
DEF_UNARY_FUNCTION(tanh)
DEF_UNARY_FUNCTION(atan)
DEF_UNARY_FUNCTION(exp)
DEF_UNARY_FUNCTION(log)
DEF_UNARY_FUNCTION(log10)
DEF_UNARY_FUNCTION(sqrt)
DEF_UNARY_FUNCTION(real)
DEF_UNARY_FUNCTION(imag)

#undef DEF_UNARY_FUNCTION

} // namespace _impl

#define _SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(_OP_,_OBJ_,_NAME_)                                                  \
	template<typename ...T1,typename  T2> _OBJ_<Expression<_impl::_NAME_,_OBJ_<T1...>,T2> >   \
	operator _OP_(_OBJ_<T1...> const & l,T2 const &r)  \
	{return (_OBJ_<Expression<_impl::_NAME_,_OBJ_<T1...>,T2> >  (l,r));}                  \


#define _SP_DEFINE_EXPR_BINARY_OPERATOR(_OP_,_OBJ_,_NAME_)                                                  \
	template<typename ...T1,typename  T2> \
	_OBJ_<Expression<_impl::_NAME_,_OBJ_<T1...>,T2> >  \
	operator _OP_(_OBJ_<T1...> const & l,T2 const &r)  \
	{return (_OBJ_<Expression<_impl::_NAME_,_OBJ_<T1...>,T2> >  (l,r));}                  \
	template< typename T1,typename ...T2> \
	_OBJ_<Expression< _impl::_NAME_,T1,_OBJ_< T2...> > > \
	operator _OP_(T1 const & l, _OBJ_< T2...>const &r)                    \
	{return (_OBJ_<Expression< _impl::_NAME_,T1,_OBJ_< T2...> > >(l,r));}                  \
	template< typename ... T1,typename ...T2> \
	_OBJ_<Expression< _impl::_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > >\
	operator _OP_(_OBJ_< T1...> const & l,_OBJ_< T2...>  const &r)                    \
	{return (_OBJ_<Expression< _impl::_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > >(l,r));}                  \

#define _SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(_OP_,_OBJ_,_NAME_)                                                  \
		template<typename ...T1,typename  T2> \
		_OBJ_<AssignmentExpression<_impl::_NAME_,_OBJ_<T1...>,T2> >  \
		operator _OP_(_OBJ_<T1...>   & l,T2 const &r)  \
		{return (_OBJ_<AssignmentExpression<_impl::_NAME_,_OBJ_<T1...>,T2> >  (l,r));}                  \

#define _SP_DEFINE_EXPR_UNARY_OPERATOR(_OP_,_OBJ_,_NAME_)                           \
		template<typename ...T> \
		_OBJ_<Expression<_impl::_NAME_,_OBJ_<T...> > >   \
		operator _OP_(_OBJ_<T...> const &l)  \
		{return (_OBJ_<Expression<_impl::_NAME_,_OBJ_<T...> > >  (l));}   \

#define _SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(_OP_,_OBJ_,_NAME_)                                                  \
	template<typename ...T1,typename  T2> \
	_OBJ_<BooleanExpression<_impl::_NAME_,_OBJ_<T1...>,T2> >  \
	operator _OP_(_OBJ_<T1...> const & l,T2 const &r)  \
	{return (_OBJ_<BooleanExpression<_impl::_NAME_,_OBJ_<T1...>,T2> >  (l,r));}                  \
	template< typename T1,typename ...T2> \
	_OBJ_<BooleanExpression< _impl::_NAME_,T1,_OBJ_< T2...> > > \
	operator _OP_(T1 const & l, _OBJ_< T2...>const &r)                    \
	{return (_OBJ_<BooleanExpression< _impl::_NAME_,T1,_OBJ_< T2...> > >(l,r));}                  \
	template< typename ... T1,typename ...T2> \
	_OBJ_<BooleanExpression< _impl::_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > >\
	operator _OP_(_OBJ_< T1...> const & l,_OBJ_< T2...>  const &r)                    \
	{return (_OBJ_<BooleanExpression< _impl::_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > >(l,r));}                  \


#define _SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR(_OP_,_OBJ_,_NAME_)                           \
		template<typename ...T> \
		_OBJ_<BooleanExpression<_impl::_NAME_,_OBJ_<T...> > >   \
		operator _OP_(_OBJ_<T...> const &l)  \
		{return (_OBJ_<BooleanExpression<_impl::_NAME_,_OBJ_<T...> > >  (l));}   \


#define _SP_DEFINE_EXPR_BINARY_FUNCTION(_NAME_,_OBJ_)                                                  \
			template<typename ...T1,typename  T2> \
			_OBJ_<Expression<_impl::_##_NAME_,_OBJ_<T1...>,T2> >   \
			_NAME_(_OBJ_<T1...> const & l,T2 const &r)  \
			{return (_OBJ_<Expression<_impl::_##_NAME_,_OBJ_<T1...>,T2> >  (l,r));}                  \
			template< typename T1,typename ...T2> \
			_OBJ_<Expression< _impl::_##_NAME_,T1,_OBJ_< T2...> > > \
			_NAME_(T1 const & l, _OBJ_< T2...>const &r)                    \
			{return (_OBJ_<Expression< _impl::_##_NAME_,T1,_OBJ_< T2...> > >(l,r));}                  \
			template< typename ... T1,typename ...T2> \
			_OBJ_<Expression< _impl::_##_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > > \
			_NAME_(_OBJ_< T1...> const & l,_OBJ_< T2...>  const &r)                    \
			{return (_OBJ_<Expression< _impl::_##_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > >(l,r));}                  \


#define _SP_DEFINE_EXPR_UNARY_FUNCTION( _NAME_,_OBJ_)                           \
		template<typename ...T> \
		_OBJ_<Expression<_impl::_##_NAME_,_OBJ_<T ...> > > \
		_NAME_(_OBJ_<T ...> const &r)  \
		{return (_OBJ_<Expression<_impl::_##_NAME_,_OBJ_<T ...> > >(r));}   \


#define  DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_CONCEPT_)          \
_SP_DEFINE_EXPR_BINARY_OPERATOR(+, _CONCEPT_, plus)                   \
_SP_DEFINE_EXPR_BINARY_OPERATOR(-, _CONCEPT_, minus)                  \
_SP_DEFINE_EXPR_BINARY_OPERATOR(*, _CONCEPT_, multiplies)             \
_SP_DEFINE_EXPR_BINARY_OPERATOR(/, _CONCEPT_, divides)                \
_SP_DEFINE_EXPR_BINARY_OPERATOR(%, _CONCEPT_, modulus)                \
_SP_DEFINE_EXPR_BINARY_OPERATOR(^, _CONCEPT_, bitwise_xor)            \
_SP_DEFINE_EXPR_BINARY_OPERATOR(&, _CONCEPT_, bitwise_and)            \
_SP_DEFINE_EXPR_BINARY_OPERATOR(|, _CONCEPT_, bitwise_or)             \
_SP_DEFINE_EXPR_UNARY_OPERATOR(~, _CONCEPT_, bitwise_not)             \
_SP_DEFINE_EXPR_UNARY_OPERATOR(+, _CONCEPT_, unary_plus)              \
_SP_DEFINE_EXPR_UNARY_OPERATOR(-, _CONCEPT_, negate)                  \
_SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(<<, _CONCEPT_, shift_left)      \
_SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(>>  , _CONCEPT_, shift_right)     \
_SP_DEFINE_EXPR_BINARY_FUNCTION(atan2, _CONCEPT_)        			  \
_SP_DEFINE_EXPR_BINARY_FUNCTION(pow, _CONCEPT_)          			  \
_SP_DEFINE_EXPR_UNARY_FUNCTION(cos, _CONCEPT_)                        \
_SP_DEFINE_EXPR_UNARY_FUNCTION(acos, _CONCEPT_)                       \
_SP_DEFINE_EXPR_UNARY_FUNCTION(cosh, _CONCEPT_)                       \
_SP_DEFINE_EXPR_UNARY_FUNCTION(sin, _CONCEPT_)                        \
_SP_DEFINE_EXPR_UNARY_FUNCTION(asin, _CONCEPT_)                       \
_SP_DEFINE_EXPR_UNARY_FUNCTION(sinh, _CONCEPT_)                       \
_SP_DEFINE_EXPR_UNARY_FUNCTION(tan, _CONCEPT_)                        \
_SP_DEFINE_EXPR_UNARY_FUNCTION(tanh, _CONCEPT_)                       \
_SP_DEFINE_EXPR_UNARY_FUNCTION(atan, _CONCEPT_)                       \
_SP_DEFINE_EXPR_UNARY_FUNCTION(exp, _CONCEPT_)                        \
_SP_DEFINE_EXPR_UNARY_FUNCTION(log, _CONCEPT_)                        \
_SP_DEFINE_EXPR_UNARY_FUNCTION(log10, _CONCEPT_)                      \
_SP_DEFINE_EXPR_UNARY_FUNCTION(sqrt, _CONCEPT_)                       \
_SP_DEFINE_EXPR_UNARY_FUNCTION(real, _CONCEPT_)                       \
_SP_DEFINE_EXPR_UNARY_FUNCTION(imag, _CONCEPT_)                       \
_SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR(!, _CONCEPT_, logical_not)     \
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(&&, _CONCEPT_, logical_and)   \
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(||, _CONCEPT_, logical_or)    \
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(!=, _CONCEPT_, not_equal_to)  \
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(==, _CONCEPT_, equal_to)      \
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(<, _CONCEPT_, less)           \
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(>, _CONCEPT_, greater)        \
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(<=, _CONCEPT_, less_equal)    \
_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(>=, _CONCEPT_, greater_equal) \
_SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(+=,_CONCEPT_, plus_assign)        \
_SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(-=,_CONCEPT_, minus_assign)       \
_SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(*=,_CONCEPT_, multiplies_assign)  \
_SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(/=,_CONCEPT_, divides_assign)     \
_SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(%=,_CONCEPT_, modulus_assign)     \


#define  DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA2(_CONCEPT_)                                              \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_OPERATOR(+, plus)                                      \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_OPERATOR(-, minus)                                     \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_OPERATOR(*, multiplies)                                \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_OPERATOR(/, divides)                                   \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_OPERATOR(%, modulus)                                   \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_OPERATOR(^, bitwise_xor)                               \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_OPERATOR(&, bitwise_and)                               \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_OPERATOR(|, bitwise_or)                                \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_OPERATOR(~, bitwise_not)                                \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_OPERATOR(+, unary_plus)                                 \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_OPERATOR(-, negate)                                     \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_RIGHT_OPERATOR(<<, shift_left)                               \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_RIGHT_OPERATOR(>>  , shift_right)                              \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_FUNCTION(atan2 )        								 \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_FUNCTION(pow)          								 \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(cos)                                           \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(acos)                                          \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(cosh)                                          \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(sin)                                           \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(asin)                                          \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(sinh)                                          \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(tan)                                           \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(tanh)                                          \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(atan)                                          \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(exp)                                           \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(log)                                           \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(log10)                                         \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(sqrt)                                          \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(real)                                          \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_FUNCTION(imag)                                          \

#define  DEFINE_EXPRESSOPM_TEMPLATE_BOOLEAN_ALGEBRA2(_CONCEPT_)                                              \
_SP_DEFINE_##_CONCEPT_##_EXPR_UNARY_BOOLEAN_OPERATOR(!,  logical_not)                              \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_BOOLEAN_OPERATOR(&&, logical_and)                              \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_BOOLEAN_OPERATOR(||, logical_or)                               \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_BOOLEAN_OPERATOR(!=, not_equal_to)                             \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_BOOLEAN_OPERATOR(==, equal_to)                                 \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_BOOLEAN_OPERATOR(<, less)                                      \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_BOOLEAN_OPERATOR(>, greater)                                   \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_BOOLEAN_OPERATOR(<=, less_equal)                               \
_SP_DEFINE_##_CONCEPT_##_EXPR_BINARY_BOOLEAN_OPERATOR(>=, greater_equal)                            \


//#undef _SP_DEFINE_EXPR_BINARY_OPERATOR
//#undef _SP_DEFINE_EXPR_UNARY_OPERATOR
//#undef _SP_DEFINE_EXPR_UNARY_FUNCTION

/** @name Constant Expresions
 * @{*/

template<typename value_type>
struct Constant
{
	value_type value;
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
struct Identity
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

//template<typename TE> inline auto operator -(Zero const &, TE const &e)
//DECL_RET_TYPE (((-e)))

inline Zero operator +(Zero const &, Zero const &e)
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

constexpr Identity operator &(Identity, Identity)
{
	return Identity();
}
template<typename TL>
TL const & operator &(TL const & l, Identity)
{
	return l;
}
template<typename TR>
TR const & operator &(Identity, TR const & r)
{
	return r;
}

template<typename TL>
constexpr Zero operator
&(TL const & l, Zero)
{
	return std::move(Zero());
}
template<typename TR>
constexpr Zero operator &(Zero, TR const & l)
{
	return std::move(Zero());
}

template<typename TR>
constexpr Zero operator &(Zero, Zero)
{
	return std::move(Zero());
}

/** @} */

/** @}*/

}   // namespace simpla

#endif /* EXPRESSION_TEMPLATE_H_ */
