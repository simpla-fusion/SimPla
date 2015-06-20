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

	_Field(this_type const & that) :
			args(that.args), m_op_(that.m_op_)
	{
	}
	_Field(this_type && that) :
			args(that.args), m_op_(that.m_op_)
	{
	}
	_Field(Args const&... pargs) :
			args(pargs ...), m_op_()
	{
	}

	_Field(TOP op, Args &&... pargs) :
			args(pargs ...), m_op_(op)
	{
	}

	~_Field()
	{
	}
//private:
//	template<typename ID, typename Tup, size_t ... index>
//	auto _invoke_helper(ID s, index_sequence<index...>)
//	DECL_RET_TYPE(m_op_(traits::index(std::get<index>(args),s)...))
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

DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_Field)

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
