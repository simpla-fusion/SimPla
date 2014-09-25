/*
 * calculus.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef CALCULUS_H_
#define CALCULUS_H_
#include <type_traits>
#include "../utilities/primitives.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/sp_functional.h"
#include "../utilities/constant_ops.h"

namespace simpla
{
template<unsigned int, typename ...> class nTuple;
template<typename ... > class Field;

template<typename TOP, typename TL, typename TR>
struct Field<TOP, TL, TR>
{
	typename StorageTraits<TL>::const_reference lhs;
	typename StorageTraits<TR>::const_reference rhs;
	TOP op_;

	Field(TL const & l, TR const & r) :
			lhs(l), rhs(r), op_()
	{
	}

	Field(TOP op, TL const & l, TR const & r) :
			lhs(l), rhs(r), op_(op)
	{
	}
	~Field()
	{
	}

	template<typename IndexType>
	inline auto operator[](IndexType const &s) const
	DECL_RET_TYPE( (op_( get_value(lhs,s),get_value(rhs,s))))
}
;

///   \brief  Unary operation
template<typename TOP, typename TL>
struct Field<TOP, TL>
{
	typename StorageTraits<TL>::const_reference lhs;
	TOP op_;

	Field(TL const & l) :
			lhs(l), op_()
	{
	}
	Field(TOP op, TL const & l) :
			lhs(l), op_(op)
	{
	}
	~Field()
	{
	}

	template<typename IndexType>
	inline auto operator[](IndexType const &s) const
	DECL_RET_TYPE( (op_( get_value(lhs,s) )))

};

template<typename TOP, typename TL, typename TR>
struct can_not_reference<Field<TOP, TL, TR>>
{
	static constexpr bool value = true;
};
/// \defgroup   BasicAlgebra Basic algebra
/// @{

#define _DEFINE_EXPR_BINARY_OPERATOR(_OP_,_NAME_)                                                  \
	template<typename TR,typename ...Args> auto operator _OP_(Field<Args...> const & l,TR const &r)  \
	DECL_RET_TYPE((Field<_impl::_NAME_,Field<Args...>,TR>(l,r)))                  \

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
_DEFINE_EXPR_BINARY_OPERATOR(==, equal_to)
_DEFINE_EXPR_BINARY_OPERATOR(!=, not_equal_to)
_DEFINE_EXPR_BINARY_OPERATOR(<, less)
_DEFINE_EXPR_BINARY_OPERATOR(>, greater)
_DEFINE_EXPR_BINARY_OPERATOR(<=, less_equal)
_DEFINE_EXPR_BINARY_OPERATOR(>=, greater_equal)
#undef _DEFINE_EXPR_BINARY_OPERATOR

#define _DEFINE_EXPR_UNARY_OPERATOR(_OP_,_NAME_)                           \
		template<typename ...Args> auto operator _OP_(Field<Args...> const &l)  \
		DECL_RET_TYPE((Field<_impl::_NAME_,Field<Args...> >(l)))   \

_DEFINE_EXPR_UNARY_OPERATOR(+, unary_plus)
_DEFINE_EXPR_UNARY_OPERATOR(-, negate)
_DEFINE_EXPR_UNARY_OPERATOR(~, bitwise_not)
#undef _DEFINE_EXPR_UNARY_OPERATOR

#define _DEFINE_EXPR_UNARY_FUNCTION( _NAME_)                           \
		template<typename ...Args> auto   _NAME_(Field<Args ...> const &r)  \
		DECL_RET_TYPE((Field<_impl::_##_NAME_,Field<Args ...>>(r)))   \

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

/// \defgroup  ExteriorAlgebra Exterior algebra
/// @{
struct HodgeStar
{
};

struct InteriorProduct
{
};

struct Wedge
{
};

struct ExteriorDerivative
{
};

struct CodifferentialDerivative
{
};

template<typename TF>
inline auto hodge_star(TF const & f)
DECL_RET_TYPE((Field<HodgeStar, TF>(f)))

template<typename TL, typename TR>
inline auto wedge(TL const & l, TR const & r)
DECL_RET_TYPE ((Field<Wedge,TL, TR>(l, r)))

template<typename TL, typename TR>
inline auto interior_product(TL const & l, TR const & r)
DECL_RET_TYPE ((Field<InteriorProduct,TL, TR>(l, r)))

template<typename TF, typename ... Others>
inline auto exterior_derivative(TF const & f, Others && ...others)
DECL_RET_TYPE((Field<ExteriorDerivative, TF>(f,
						std::forward<Others>(others)...)))

template<typename TF, typename ... Others>
inline auto codifferential_derivative(TF const & f, Others && ...others)
DECL_RET_TYPE((Field<CodifferentialDerivative, TF>(f,
						std::forward<Others>(others)...)))

template<typename TD, typename TL>
inline auto operator*(Field<TD, TL> const & f)
DECL_RET_TYPE((hodge_star(f)))
;
template<typename TD, typename TL>
inline auto d(Field<TD, TL> const & f)
DECL_RET_TYPE( (exterior_derivative(f)) )
;

template<typename TD, typename TL>
inline auto delta(Field<TD, TL> const & f)
DECL_RET_TYPE( (codifferential_derivative(f)) )
;

template<typename TD, typename TL, typename TR>
inline auto iv(nTuple<TD::NDIMS, TL> const & v, Field<TD, TR> const & f)
DECL_RET_TYPE( (interior_product(v,f)) )
;

template<typename TDL, typename TL, typename TDR, typename TR>
inline auto operator^(Field<TDL, TL> const & lhs, Field<TDR, TR> const & rhs)
DECL_RET_TYPE( (wedge(lhs,rhs)) )
;

///  @}

///  \defgroup  VectorAlgebra Vector algebra
///  @{
template<typename TL, typename TR> inline auto InnerProduct(TL const & lhs,
		TR const & rhs)
		DECL_RET_TYPE(wedge (lhs,hodge_star( rhs) ))
;

template<typename TL, typename TR> inline auto Dot(TL const & lhs,
		TR const & rhs)
		DECL_RET_TYPE(wedge(lhs , rhs ))
;
template<typename TL, typename TR> inline auto Cross(TL const & lhs,
		TR const & rhs)
		DECL_RET_TYPE( wedge(lhs , rhs ))
;

template<typename TL, typename TR> inline auto Dot(nTuple<3, TL> const & v,
		TR const & f)
		DECL_RET_TYPE( (interior_product(v, f)))
;

template<typename TL, typename TR> inline auto Dot(TL const & f,
		nTuple<3, TR> const & v)
		DECL_RET_TYPE( (interior_product(v, f)))
;

template<typename TL, typename TR> inline auto Cross(TL const & f,
		nTuple<3, TR> const & v)
		DECL_RET_TYPE( (interior_product(v, hodge_star(f))))
;

template<typename TM, typename TL, typename TR> inline auto Cross(
		Field<Domain<TM, FACE>, TR> const & f, nTuple<3, TL> const & v)
		DECL_RET_TYPE((interior_product(v, f)))
;

template<typename TF>
inline auto grad(TF const & f)
DECL_RET_TYPE( ( exterior_derivative(f)))
;

template<typename TF>
inline auto diverge(TF const & f)
DECL_RET_TYPE((exterior_derivative( f)))
;

template<typename TM, typename TR>
inline auto curl(Field<Domain<TM, EDGE>, TR> const & f)
DECL_RET_TYPE((exterior_derivative(f)))
;

template<typename TM, typename TR>
inline auto grad(Field<Domain<TM, VOLUME>, TR> const & f)
DECL_RET_TYPE(-(codifferential(f)))
;

template<typename TM, typename TR>
inline auto diverge(Field<Domain<TM, EDGE>, TR> const & f)
DECL_RET_TYPE(-(codifferential_derivative(f)))
;

template<typename TM, typename TR>
inline auto curl(Field<Domain<TM, FACE>, TR> const & f)
DECL_RET_TYPE(-(codifferential_derivative(f)))
;

///   @}

///  \ingroup  FETL
///  \defgroup  NonstandardOperations Non-standard operations
///   @{
template<typename TM, typename TR>
inline auto CurlPDX(
		Field<Domain<TM, EDGE>, TR> const & f)
				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<unsigned int ,0>())))
;

template<typename TM, typename TR>
inline auto CurlPDY(
		Field<Domain<TM, EDGE>, TR> const & f)
				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<unsigned int ,1>())))
;

template<typename TM, typename TR>
inline auto CurlPDZ(
		Field<Domain<TM, EDGE>, TR> const & f)
				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<unsigned int ,2>())))
;

template<typename TM, typename TR>
inline auto CurlPDX(
		Field<Domain<TM, FACE>, TR> const & f)
				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<unsigned int ,0>())))
;

template<typename TM, typename TR>
inline auto CurlPDY(
		Field<Domain<TM, FACE>, TR> const & f)
				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<unsigned int ,1>())))
;

template<typename TM, typename TR>
inline auto CurlPDZ(
		Field<Domain<TM, FACE>, TR> const & f)
				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<unsigned int ,2>())))
;

//template<unsigned int IL, typename TM, unsigned int IR, typename TR>
//inline auto MapTo(
//		Field<Domain<TM, IR>, TR> const & f)
//				DECL_RET_TYPE( (Field<Domain<TM,IL>, Field<MAPTO,std::integral_constant<unsigned int ,IL>,Field<Domain<TM, IR>, TR> > >(std::integral_constant<unsigned int ,IL>(), f)))
//;
//
//template<unsigned int IL, typename TM, unsigned int IR, typename TR>
//inline auto MapTo(std::integral_constant<unsigned int, IL>,
//		Field<Domain<TM, IR>, TR> const & f)
//				DECL_RET_TYPE( (Field<Domain<TM,IL>, Field<MAPTO,std::integral_constant<unsigned int ,IL>,Field<Domain<TM, IR>, TR> > >(std::integral_constant<unsigned int ,IL>(), f)))
//;

///   @}

}
// namespace simpla

#endif /* CALCULUS_H_ */
