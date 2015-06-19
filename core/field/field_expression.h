/**
 * @file field_expression.h
 *
 *  Created on: 2015年1月30日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_EXPRESSION_H_
#define CORE_FIELD_FIELD_EXPRESSION_H_
#include "../gtl/expression_template.h"
#include "../gtl/type_traits.h"
#include "../utilities/log.h"
#include "field_traits.h"
namespace simpla
{
/** @addtogroup field
 *  @{
 */
template<typename ... >struct _Field;
template<typename ...> struct Domain;

/// @name  Field Expression
/// @{

template<typename ...>class Expression;
template<typename ...>class BooleanExpression;

template<typename TOP, typename ...Args>
struct _Field<Expression<TOP, Args...> >
{
	typedef _Field<Expression<TOP, Args...> > this_type;

	typename std::tuple<traits::reference_t<Args> ...> args;

	TOP m_op_;

	_Field(this_type const & that)
			: args(that.args), m_op_(that.m_op_)
	{
	}
	_Field(this_type && that)
			: args(that.args), m_op_(that.m_op_)
	{
	}
	_Field(Args const&... pargs)
			: args(pargs ...), m_op_()
	{
	}

	_Field(TOP op, Args &&... pargs)
			: args(pargs ...), m_op_(op)
	{
	}

	~_Field()
	{
	}
//private:
//	template<typename ID, typename Tup, size_t ... index>
//	auto _invoke_helper(ID s, index_sequence<index...>)
//	DECL_RET_TYPE(m_op_(try_index(std::get<index>(args),s)...))
//
//public:
//	template<typename ID>
//	auto at(
//			ID const &s)
//					DECL_RET_TYPE((
//									_invoke_helper( s ,
//											typename make_index_sequence<sizeof...(Args)>::type () )))
//
//	template<typename ID>
//	inline auto operator[](ID const &s) const
//	DECL_RET_TYPE ( at(s))

}
;

template<typename ...T>
struct _Field<BooleanExpression<T...> > : public _Field<Expression<T...>>
{
	using _Field<Expression<T...> >::_Field;
};
namespace traits
{

template<typename > struct value_type;
template<typename > struct field_value_type;

template<typename TOP, typename ...T>
struct value_type<_Field<BooleanExpression<TOP, T...> > >
{
	typedef bool type;
};
template<typename TOP, typename ...T>
struct field_value_type<_Field<BooleanExpression<TOP, T...> > >
{
	typedef bool type;
};

template<typename TOP, typename ...T>
struct value_type<_Field<Expression<TOP, T...> > >
{
	typedef result_of_t<TOP(value_type_t<T> ...)> type;
};

namespace _impl
{
template<typename ...T> struct first_field;

template<typename ...T> using first_field_t=typename first_field<T...>::type;

template<typename T0> struct first_field<T0>
{
	typedef domain_t<T0> type;
};
template<typename T0, typename ...T> struct first_field<T0, T...>
{
	typedef typename std::conditional<is_field<T0>::value,
			typename is_field<T...>::type, first_domain_t<T0> >::type type;
};

}  // namespace _impl

template<typename TAG, typename T0, typename ... T>
struct iform<_Field<Expression<TAG, T0, T...> > > : public traits::iform<T0>::type
{
}
;
}  // namespace traits

template<typename TOP, typename TL, typename TR>
struct _Field<AssignmentExpression<TOP, TL, TR>> : public AssignmentExpression<
		TOP, TL, TR>
{
	typedef AssignmentExpression<TOP, TL, TR> expression_type;

	typedef traits::value_type_t<TL> value_type;

	typedef traits::domain_t<TL> domain_type;

	typedef _Field<AssignmentExpression<TOP, TL, TR>> this_type;

	using AssignmentExpression<TOP, TL, TR>::AssignmentExpression;

	bool is_excuted_ = false;

	void excute()
	{
		if (!is_excuted_)
		{
			expression_type::lhs.mesh().calculate(*this);
			is_excuted_ = true;
		}

	}
	void do_not_excute()
	{
		is_excuted_ = true;
	}

	~_Field()
	{
		excute();
	}
};

//DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_Field)

/////////////////////////////////////////////////////////////////////

template<typename ...T1, typename T2>
_Field<Expression<_impl::plus, _Field<T1...>, T2> > operator +(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::plus, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<Expression<_impl::plus, T1, _Field<T2...> > > operator +(T1 const & l,
		_Field<T2...> const &r)
{
	return (_Field<Expression<_impl::plus, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<Expression<_impl::plus, _Field<T1...>, _Field<T2...> > >\
 operator +(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::plus, _Field<T1...>, _Field<T2...> > >(l,
			r));
}
template<typename ...T1, typename T2>
_Field<Expression<_impl::minus, _Field<T1...>, T2> > operator -(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::minus, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<Expression<_impl::minus, T1, _Field<T2...> > > operator -(T1 const & l,
		_Field<T2...> const &r)
{
	return (_Field<Expression<_impl::minus, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<Expression<_impl::minus, _Field<T1...>, _Field<T2...> > >\
 operator -(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::minus, _Field<T1...>, _Field<T2...> > >(l,
			r));
}
template<typename ...T1, typename T2>
_Field<Expression<_impl::multiplies, _Field<T1...>, T2> > operator *(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::multiplies, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<Expression<_impl::multiplies, T1, _Field<T2...> > > operator *(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::multiplies, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<Expression<_impl::multiplies, _Field<T1...>, _Field<T2...> > >\
 operator *(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::multiplies, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<Expression<_impl::divides, _Field<T1...>, T2> > operator /(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::divides, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<Expression<_impl::divides, T1, _Field<T2...> > > operator /(T1 const & l,
		_Field<T2...> const &r)
{
	return (_Field<Expression<_impl::divides, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<Expression<_impl::divides, _Field<T1...>, _Field<T2...> > >\
 operator /(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::divides, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<Expression<_impl::modulus, _Field<T1...>, T2> > operator %(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::modulus, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<Expression<_impl::modulus, T1, _Field<T2...> > > operator %(T1 const & l,
		_Field<T2...> const &r)
{
	return (_Field<Expression<_impl::modulus, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<Expression<_impl::modulus, _Field<T1...>, _Field<T2...> > >\
 operator %(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::modulus, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<Expression<_impl::bitwise_xor, _Field<T1...>, T2> > operator ^(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::bitwise_xor, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<Expression<_impl::bitwise_xor, T1, _Field<T2...> > > operator ^(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::bitwise_xor, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<Expression<_impl::bitwise_xor, _Field<T1...>, _Field<T2...> > >\
 operator ^(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::bitwise_xor, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<Expression<_impl::bitwise_and, _Field<T1...>, T2> > operator &(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::bitwise_and, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<Expression<_impl::bitwise_and, T1, _Field<T2...> > > operator &(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::bitwise_and, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<Expression<_impl::bitwise_and, _Field<T1...>, _Field<T2...> > >\
 operator &(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::bitwise_and, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<Expression<_impl::bitwise_or, _Field<T1...>, T2> > operator |(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::bitwise_or, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<Expression<_impl::bitwise_or, T1, _Field<T2...> > > operator |(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::bitwise_or, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<Expression<_impl::bitwise_or, _Field<T1...>, _Field<T2...> > >\
 operator |(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::bitwise_or, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T>
_Field<Expression<_impl::bitwise_not, _Field<T...> > > operator ~(
		_Field<T...> const &l)
{
	return (_Field<Expression<_impl::bitwise_not, _Field<T...> > >(l));
}
template<typename ...T>
_Field<Expression<_impl::unary_plus, _Field<T...> > > operator +(
		_Field<T...> const &l)
{
	return (_Field<Expression<_impl::unary_plus, _Field<T...> > >(l));
}
template<typename ...T>
_Field<Expression<_impl::negate, _Field<T...> > > operator -(
		_Field<T...> const &l)
{
	return (_Field<Expression<_impl::negate, _Field<T...> > >(l));
}
template<typename ...T1, typename T2> _Field<
		Expression<_impl::shift_left, _Field<T1...>, T2> > operator <<(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::shift_left, _Field<T1...>, T2> >(l, r));
}
template<typename ...T1, typename T2> _Field<
		Expression<_impl::shift_right, _Field<T1...>, T2> > operator >>(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::shift_right, _Field<T1...>, T2> >(l, r));
}
template<typename ...T1, typename T2>
_Field<Expression<_impl::_atan2, _Field<T1...>, T2> > atan2(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<Expression<_impl::_atan2, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<Expression<_impl::_atan2, T1, _Field<T2...> > > atan2(T1 const & l,
		_Field<T2...> const &r)
{
	return (_Field<Expression<_impl::_atan2, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<Expression<_impl::_atan2, _Field<T1...>, _Field<T2...> > > atan2(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::_atan2, _Field<T1...>, _Field<T2...> > >(l,
			r));
}
template<typename ...T1, typename T2>
_Field<Expression<_impl::_pow, _Field<T1...>, T2> > pow(_Field<T1...> const & l,
		T2 const &r)
{
	return (_Field<Expression<_impl::_pow, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<Expression<_impl::_pow, T1, _Field<T2...> > > pow(T1 const & l,
		_Field<T2...> const &r)
{
	return (_Field<Expression<_impl::_pow, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<Expression<_impl::_pow, _Field<T1...>, _Field<T2...> > > pow(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<Expression<_impl::_pow, _Field<T1...>, _Field<T2...> > >(l,
			r));
}
template<typename ...T>
_Field<Expression<_impl::_cos, _Field<T ...> > > cos(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_cos, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_acos, _Field<T ...> > > acos(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_acos, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_cosh, _Field<T ...> > > cosh(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_cosh, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_sin, _Field<T ...> > > sin(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_sin, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_asin, _Field<T ...> > > asin(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_asin, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_sinh, _Field<T ...> > > sinh(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_sinh, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_tan, _Field<T ...> > > tan(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_tan, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_tanh, _Field<T ...> > > tanh(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_tanh, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_atan, _Field<T ...> > > atan(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_atan, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_exp, _Field<T ...> > > exp(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_exp, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_log, _Field<T ...> > > log(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_log, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_log10, _Field<T ...> > > log10(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_log10, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_sqrt, _Field<T ...> > > sqrt(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_sqrt, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_real, _Field<T ...> > > real(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_real, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<Expression<_impl::_imag, _Field<T ...> > > imag(_Field<T ...> const &r)
{
	return (_Field<Expression<_impl::_imag, _Field<T ...> > >(r));
}
template<typename ...T>
_Field<BooleanExpression<_impl::logical_not, _Field<T...> > > operator !(
		_Field<T...> const &l)
{
	return (_Field<BooleanExpression<_impl::logical_not, _Field<T...> > >(l));
}
template<typename ...T1, typename T2>
_Field<BooleanExpression<_impl::logical_and, _Field<T1...>, T2> > operator &&(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<BooleanExpression<_impl::logical_and, _Field<T1...>, T2> >(l,
			r));
}
template<typename T1, typename ...T2>
_Field<BooleanExpression<_impl::logical_and, T1, _Field<T2...> > > operator &&(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<BooleanExpression<_impl::logical_and, T1, _Field<T2...> > >(
			l, r));
}
template<typename ... T1, typename ...T2>
_Field<BooleanExpression<_impl::logical_and, _Field<T1...>, _Field<T2...> > >\
 operator &&(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<
			BooleanExpression<_impl::logical_and, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<BooleanExpression<_impl::logical_or, _Field<T1...>, T2> > operator ||(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<BooleanExpression<_impl::logical_or, _Field<T1...>, T2> >(l,
			r));
}
template<typename T1, typename ...T2>
_Field<BooleanExpression<_impl::logical_or, T1, _Field<T2...> > > operator ||(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<BooleanExpression<_impl::logical_or, T1, _Field<T2...> > >(l,
			r));
}
template<typename ... T1, typename ...T2>
_Field<BooleanExpression<_impl::logical_or, _Field<T1...>, _Field<T2...> > >\
 operator ||(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<
			BooleanExpression<_impl::logical_or, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<BooleanExpression<_impl::not_equal_to, _Field<T1...>, T2> > operator !=(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<BooleanExpression<_impl::not_equal_to, _Field<T1...>, T2> >(
			l, r));
}
template<typename T1, typename ...T2>
_Field<BooleanExpression<_impl::not_equal_to, T1, _Field<T2...> > > operator !=(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<BooleanExpression<_impl::not_equal_to, T1, _Field<T2...> > >(
			l, r));
}
template<typename ... T1, typename ...T2>
_Field<BooleanExpression<_impl::not_equal_to, _Field<T1...>, _Field<T2...> > >\
 operator !=(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<
			BooleanExpression<_impl::not_equal_to, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<BooleanExpression<_impl::equal_to, _Field<T1...>, T2> > operator ==(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<BooleanExpression<_impl::equal_to, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<BooleanExpression<_impl::equal_to, T1, _Field<T2...> > > operator ==(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<BooleanExpression<_impl::equal_to, T1, _Field<T2...> > >(l,
			r));
}
template<typename ... T1, typename ...T2>
_Field<BooleanExpression<_impl::equal_to, _Field<T1...>, _Field<T2...> > >\
 operator ==(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<
			BooleanExpression<_impl::equal_to, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<BooleanExpression<_impl::less, _Field<T1...>, T2> > operator <(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<BooleanExpression<_impl::less, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<BooleanExpression<_impl::less, T1, _Field<T2...> > > operator <(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<BooleanExpression<_impl::less, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<BooleanExpression<_impl::less, _Field<T1...>, _Field<T2...> > >\
 operator <(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<BooleanExpression<_impl::less, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<BooleanExpression<_impl::greater, _Field<T1...>, T2> > operator >(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<BooleanExpression<_impl::greater, _Field<T1...>, T2> >(l, r));
}
template<typename T1, typename ...T2>
_Field<BooleanExpression<_impl::greater, T1, _Field<T2...> > > operator >(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<BooleanExpression<_impl::greater, T1, _Field<T2...> > >(l, r));
}
template<typename ... T1, typename ...T2>
_Field<BooleanExpression<_impl::greater, _Field<T1...>, _Field<T2...> > >\
 operator >(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<
			BooleanExpression<_impl::greater, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<BooleanExpression<_impl::less_equal, _Field<T1...>, T2> > operator <=(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<BooleanExpression<_impl::less_equal, _Field<T1...>, T2> >(l,
			r));
}
template<typename T1, typename ...T2>
_Field<BooleanExpression<_impl::less_equal, T1, _Field<T2...> > > operator <=(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<BooleanExpression<_impl::less_equal, T1, _Field<T2...> > >(l,
			r));
}
template<typename ... T1, typename ...T2>
_Field<BooleanExpression<_impl::less_equal, _Field<T1...>, _Field<T2...> > >\
 operator <=(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<
			BooleanExpression<_impl::less_equal, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<BooleanExpression<_impl::greater_equal, _Field<T1...>, T2> > operator >=(
		_Field<T1...> const & l, T2 const &r)
{
	return (_Field<BooleanExpression<_impl::greater_equal, _Field<T1...>, T2> >(
			l, r));
}
template<typename T1, typename ...T2>
_Field<BooleanExpression<_impl::greater_equal, T1, _Field<T2...> > > operator >=(
		T1 const & l, _Field<T2...> const &r)
{
	return (_Field<BooleanExpression<_impl::greater_equal, T1, _Field<T2...> > >(
			l, r));
}
template<typename ... T1, typename ...T2>
_Field<BooleanExpression<_impl::greater_equal, _Field<T1...>, _Field<T2...> > >\
 operator >=(
		_Field<T1...> const & l, _Field<T2...> const &r)
{
	return (_Field<
			BooleanExpression<_impl::greater_equal, _Field<T1...>, _Field<T2...> > >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<AssignmentExpression<_impl::plus_assign, _Field<T1...>, T2> > operator +=(
		_Field<T1...> & l, T2 const &r)
{
	return (_Field<AssignmentExpression<_impl::plus_assign, _Field<T1...>, T2> >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<AssignmentExpression<_impl::minus_assign, _Field<T1...>, T2> > operator -=(
		_Field<T1...> & l, T2 const &r)
{
	return (_Field<AssignmentExpression<_impl::minus_assign, _Field<T1...>, T2> >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<AssignmentExpression<_impl::multiplies_assign, _Field<T1...>, T2> > operator *=(
		_Field<T1...> & l, T2 const &r)
{
	return (_Field<
			AssignmentExpression<_impl::multiplies_assign, _Field<T1...>, T2> >(
			l, r));
}
template<typename ...T1, typename T2>
_Field<AssignmentExpression<_impl::divides_assign, _Field<T1...>, T2> > operator /=(
		_Field<T1...> & l, T2 const &r)
{
	return (_Field<
			AssignmentExpression<_impl::divides_assign, _Field<T1...>, T2> >(l,
			r));
}
template<typename ...T1, typename T2>
_Field<AssignmentExpression<_impl::modulus_assign, _Field<T1...>, T2> > operator %=(
		_Field<T1...> & l, T2 const &r)
{
	return (_Field<
			AssignmentExpression<_impl::modulus_assign, _Field<T1...>, T2> >(l,
			r));
}
/////////////////////////////////////////////////////////////////////

#define SP_DEF_BINOP_FIELD_NTUPLE(_OP_,_NAME_)                                                 \
template<typename ...T1, typename T2, size_t ... N>                                            \
_Field<Expression<_impl::plus, _Field<T1...>, nTuple<T2, N...> > > operator _OP_(              \
		_Field<T1...> const & l, nTuple<T2, N...> const &r)                                    \
{return (_Field<Expression<_impl::_NAME_, _Field<T1...>, nTuple<T2, N...> > >(l, r));}         \
template<typename T1, size_t ... N, typename ...T2>                                            \
_Field<Expression<_impl::plus, nTuple<T1, N...>, _Field<T2...> > > operator _OP_(              \
		nTuple<T1, N...> const & l, _Field< T2...>const &r)                                    \
{	return (_Field<Expression< _impl::_NAME_,T1,_Field< T2...>>>(l,r));}                       \


SP_DEF_BINOP_FIELD_NTUPLE(+, plus)
SP_DEF_BINOP_FIELD_NTUPLE(-, minus)
SP_DEF_BINOP_FIELD_NTUPLE(*, multiplies)
SP_DEF_BINOP_FIELD_NTUPLE(/, divides)
SP_DEF_BINOP_FIELD_NTUPLE(%, modulus)
SP_DEF_BINOP_FIELD_NTUPLE(^, bitwise_xor)
SP_DEF_BINOP_FIELD_NTUPLE(&, bitwise_and)
SP_DEF_BINOP_FIELD_NTUPLE(|, bitwise_or)
#undef SP_DEF_BINOP_FIELD_NTUPLE

/** @} */
}  // namespace simpla

#endif /* CORE_FIELD_FIELD_EXPRESSION_H_ */
