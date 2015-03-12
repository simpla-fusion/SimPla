/**
 * @file field_expression.h
 *
 *  Created on: 2015年1月30日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_EXPRESSION_H_
#define CORE_FIELD_FIELD_EXPRESSION_H_
#include "../gtl/expression_template.h"
#include "../utilities/log.h"
namespace simpla
{
/** @addtogroup field
 *  @{
 */
template<typename ... >struct _Field;

namespace _impl
{
class is_sequence_container;

class is_associative_container;

}
// namespace _impl

template<typename TM, typename TV, typename ...Others>
struct reference_traits<_Field<TM, TV, Others...> >
{
	typedef _Field<TM, TV, Others...> const & type;
};

template<typename > struct is_field
{
	static constexpr bool value = false;
};

template<typename ...T> struct is_field<_Field<T...>>
{
	static constexpr bool value = true;
};

template<typename ...> struct field_traits;

template<typename T> struct field_traits<T>
{

	typedef std::nullptr_t mesh_type;

	typedef T value_type;

	static constexpr bool is_field = false;

};

template<typename ...T>
struct field_traits<_Field<T ...>>
{
	static constexpr bool is_field = true;

	typedef typename _Field<T ...>::mesh_type mesh_type;

	typedef typename _Field<T ...>::value_type value_type;

	static constexpr size_t iform = mesh_type::iform;

	static constexpr size_t ndims = mesh_type::ndims;

};

/// @name  Field Expression
/// @{

template<typename ...>class Expression;
template<typename ...>class BooleanExpression;

template<typename TOP, typename TL>
struct _Field<Expression<TOP, TL, std::nullptr_t>> : public Expression<TOP, TL,
		std::nullptr_t>
{
	typedef typename field_traits<TL>::value_type l_type;
public:

	typedef typename field_traits<TL>::mesh_type mesh_type;

	typedef typename sp_result_of<TOP(l_type)>::type value_type;

	typedef _Field<Expression<TOP, TL, std::nullptr_t>> this_type;

	using Expression<TOP, TL, std::nullptr_t>::Expression;
};

template<typename TOP, typename TL, typename TR>
struct _Field<Expression<TOP, TL, TR>> : public Expression<TOP, TL, TR>
{

	typedef typename field_traits<TL>::value_type l_type;
	typedef typename field_traits<TR>::value_type r_type;

public:

	typedef typename sp_result_of<TOP(l_type, r_type)>::type value_type;

	typedef typename field_traits<TL>::mesh_type mesh_type;

	typedef _Field<Expression<TOP, TL, TR>> this_type;

	using Expression<TOP, TL, TR>::Expression;
};

template<typename TOP, typename TL, typename TR>
struct _Field<BooleanExpression<TOP, TL, TR>> : public Expression<TOP, TL, TR>
{
	typedef bool value_type;

	typedef typename field_traits<TL>::mesh_type mesh_type;

	typedef _Field<BooleanExpression<TOP, TL, TR>> this_type;

	using Expression<TOP, TL, TR>::Expression;

	operator bool() const
	{
		UNIMPLEMENTED;
		return false;
	}
};

template<typename TOP, typename TL, typename TR>
struct _Field<AssignmentExpression<TOP, TL, TR>> : public AssignmentExpression<
		TOP, TL, TR>
{
	typedef AssignmentExpression<TOP, TL, TR> expression_type;

	typedef typename field_traits<TL>::value_type value_type;

	typedef typename field_traits<TL>::mesh_type mesh_type;

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
