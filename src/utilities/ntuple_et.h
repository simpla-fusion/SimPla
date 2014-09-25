/*
 * ntuple_et.h
 *
 *  created on: 2014-4-1
 *      Author: salmon
 */

#ifndef NTUPLE_ET_H_
#define NTUPLE_ET_H_

#include <stddef.h>

#include "sp_functional.h"
#include "sp_type_traits.h"

namespace simpla
{
// Expression template of nTuple
template<unsigned int N, typename ... > struct nTuple;
template<unsigned int N, typename T> using Matrix=nTuple<N,nTuple<N,T>>;
//***********************************************************************************
template<unsigned int N, typename TOP, typename TL>
struct nTuple<N, TOP, TL>
{
	typename StorageTraits<TL>::const_reference lhs;
	TOP op_;
	nTuple(TL const & l) :
			lhs(l)
	{
	}
	nTuple(TOP op, TL const & l) :
			lhs(l), op_(op)
	{
	}
	~nTuple()
	{
	}
	template<typename TI>
	constexpr auto operator[](TI s) const
	DECL_RET_TYPE(op_( get_value(lhs,s) ))

	template<typename T>
	inline operator nTuple<N,T>() const
	{
		nTuple<N, T> res;

		_ntuple_impl::assign<N>(_impl::binary_right(), res, *this);

		return (res);
	}

};
template<unsigned int N, typename TOP, typename TL, typename TR>
struct nTuple<N, TOP, TL, TR>
{
	typename StorageTraits<TL>::const_reference lhs;
	typename StorageTraits<TR>::const_reference rhs;
	TOP op_;
	nTuple(TL const & l, TR const & r) :
			lhs(l), rhs(r)
	{
	}
	nTuple(TOP op, TL const & l, TR const & r) :
			lhs(l), rhs(r), op_(op)
	{
	}
	~nTuple()
	{
	}
	template<typename TI>
	constexpr auto operator[](TI s) const
	DECL_RET_TYPE(op_( get_value(lhs,s),get_value(rhs,s)))

	template<typename T>
	inline operator nTuple<N,T>() const
	{
		nTuple<N, T> res;

		_ntuple_impl::assign<N>(_impl::binary_right(), res, *this);

		return (res);
	}

};
template<unsigned int N, typename TOP, typename ... TL>
struct can_not_reference<nTuple<N, TOP, TL ...>>
{
	static constexpr bool value = true;
};

/// \defgroup   BasicAlgebra Basic algebra
/// @{

#define _DEFINE_EXPR_BINARY_OPERATOR(_OP_,_NAME_)                                                  \
	template<unsigned int N,typename TR,typename ...Args> auto operator _OP_(nTuple<N,Args...> const & l,TR const &r)                    \
	DECL_RET_TYPE((nTuple<N,_impl::_NAME_,nTuple<N,Args...>,TR>(l,r)))                  \
	template<unsigned int N,typename TR,typename ...Args> auto operator _OP_(TR const & l, nTuple<N,Args...>const &r)                    \
	DECL_RET_TYPE((nTuple<N,_impl::_NAME_,TR,nTuple<N,Args...>>(l,r)))                  \
	template<unsigned int N,typename ... TL,typename ...TR> auto operator _OP_(nTuple<N,TL...> const & l,nTuple<N,TR...>  const &r)                    \
	DECL_RET_TYPE((nTuple<N,_impl::_NAME_,nTuple<N,TL...>,nTuple<N,TR...>>(l,r)))                  \

_DEFINE_EXPR_BINARY_OPERATOR(+, plus)
_DEFINE_EXPR_BINARY_OPERATOR(-, minus)
_DEFINE_EXPR_BINARY_OPERATOR(*, multiplies)
_DEFINE_EXPR_BINARY_OPERATOR(/, divides)
_DEFINE_EXPR_BINARY_OPERATOR(%, modulus)
_DEFINE_EXPR_BINARY_OPERATOR(^, bitwise_xor)
_DEFINE_EXPR_BINARY_OPERATOR(&, bitwise_and)
_DEFINE_EXPR_BINARY_OPERATOR(|, bitwise_or)
_DEFINE_EXPR_BINARY_OPERATOR(<<, shift_left)
_DEFINE_EXPR_BINARY_OPERATOR(>>, shift_right)
_DEFINE_EXPR_BINARY_OPERATOR(&&, logical_and)
_DEFINE_EXPR_BINARY_OPERATOR(||, logical_or)
//_DEFINE_EXPR_BINARY_OPERATOR(==, equal_to)
//_DEFINE_EXPR_BINARY_OPERATOR(!=, not_equal_to)
_DEFINE_EXPR_BINARY_OPERATOR(<, less)
_DEFINE_EXPR_BINARY_OPERATOR(>, greater)
_DEFINE_EXPR_BINARY_OPERATOR(<=, less_equal)
_DEFINE_EXPR_BINARY_OPERATOR(>=, greater_equal)
#undef _DEFINE_EXPR_BINARY_OPERATOR

#define _DEFINE_EXPR_UNARY_OPERATOR(_OP_,_NAME_)                           \
		template<unsigned int N,typename ...Args> auto operator _OP_(nTuple<N,Args...> const &l)                    \
		DECL_RET_TYPE((nTuple<N,_impl::_NAME_,nTuple<N,Args...> >(l)))   \

_DEFINE_EXPR_UNARY_OPERATOR(+, unary_plus)
_DEFINE_EXPR_UNARY_OPERATOR(-, negate)
_DEFINE_EXPR_UNARY_OPERATOR(~, bitwise_not)
#undef _DEFINE_EXPR_UNARY_OPERATOR

#define _DEFINE_EXPR_UNARY_FUNCTION( _NAME_)                           \
		template<unsigned int N,typename ...Args> auto   _NAME_(nTuple<N,Args ...> const &r)                    \
		DECL_RET_TYPE((nTuple<N,_impl::_##_NAME_,nTuple<N,Args ...> >(r)))   \

_DEFINE_EXPR_UNARY_FUNCTION(abs)
_DEFINE_EXPR_UNARY_FUNCTION(cos)
_DEFINE_EXPR_UNARY_FUNCTION(acos)
_DEFINE_EXPR_UNARY_FUNCTION(cosh)
_DEFINE_EXPR_UNARY_FUNCTION(sin)
_DEFINE_EXPR_UNARY_FUNCTION(asin)
_DEFINE_EXPR_UNARY_FUNCTION(sinh)
_DEFINE_EXPR_UNARY_FUNCTION(tan)
_DEFINE_EXPR_UNARY_FUNCTION(tanh)
_DEFINE_EXPR_UNARY_FUNCTION(atan)
_DEFINE_EXPR_UNARY_FUNCTION(exp)
_DEFINE_EXPR_UNARY_FUNCTION(log)
_DEFINE_EXPR_UNARY_FUNCTION(log10)
_DEFINE_EXPR_UNARY_FUNCTION(sqrt)
_DEFINE_EXPR_UNARY_FUNCTION(real)
_DEFINE_EXPR_UNARY_FUNCTION(imag)
#undef _DEFINE_EXPR_UNARY_FUNCTION
/// @}

//***********************************************************************************
namespace ntuple_impl
{

template<unsigned int M, typename TL, typename TR> struct _inner_product_s;

template<typename TL, typename TR>
inline auto _inner_product(TL const & l, TR const &r)
DECL_RET_TYPE((l*r))

template<unsigned int N, typename TL, typename TR>
inline auto _inner_product(nTuple<N, TL> const & l,
		nTuple<N, TR> const &r)
				DECL_RET_TYPE(( _inner_product_s<N, nTuple<N, TL>, nTuple<N, TR> >::calculus(l,r)))

template<unsigned int M, typename TL, typename TR>
struct _inner_product_s
{
	static inline auto calculus(TL const & l,
			TR const &r)
					DECL_RET_TYPE((_inner_product(l[M - 1] , r[M - 1]) + _inner_product_s<M - 1, TL, TR>::calculus(l, r)))
};
template<typename TL, typename TR>
struct _inner_product_s<1, TL, TR>
{
	static inline auto calculus(TL const & l, TR const &r)
	DECL_RET_TYPE(_inner_product(l[0],r[0]))
}
;

}
//namespace ntuple_impl
template<unsigned int N, typename TL, typename TR>
inline auto dot(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE((ntuple_impl::_inner_product(l,r)))

template<unsigned int N, typename TL, typename TR>
inline auto inner_product(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE((ntuple_impl::_inner_product(l,r)))

template<typename TR, typename ...Args> inline auto cross(
		nTuple<3, Args...> const & l, TR const & r)
		->nTuple<3,decltype(l[0]*r[0])>
{
	nTuple<3, decltype(l[0]*r[0])> res =
	{ l[1] * r[2] - l[2] * r[1], l[2] * r[0] - l[0] * r[2], l[0] * r[1]
			- l[1] * r[0] };
	return std::move(res);
}
//***********************************************************************************
// overloading operators

}// namespace simpla
#endif /* NTUPLE_ET_H_ */
