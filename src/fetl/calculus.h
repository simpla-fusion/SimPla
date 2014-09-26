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
#include "../utilities/ntuple.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/sp_functional.h"
#include "../utilities/constant_ops.h"
#include "../utilities/expression_template.h"

namespace simpla
{
template<typename ... > class Field;
template<typename ... > class Expression;

template<typename ... T>
class Field<Expression<T...>> : public Expression<T...>
{
	using Expression<T...>::Expression;
};

/// \defgroup   BasicAlgebra Basic algebra
/// @{

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
inline Field<Expression<HodgeStar, TF>> hodge_star(TF const & f)
{
	return std::move((Field<Expression<HodgeStar, TF>>(f)));
}
//DECL_RET_TYPE((Field<Expression<HodgeStar, TF>>(f)))

template<typename TL, typename TR>
inline Field<Expression<Wedge, TL, TR>> wedge(TL const & l, TR const & r)
{
	return std::move(Field<Expression<Wedge, TL, TR>>(l, r));
}

template<typename TL, typename TR>
inline auto interior_product(TL const & l, TR const & r)
DECL_RET_TYPE ((Field<Expression<InteriorProduct,TL, TR>>(l, r)))

template<typename TF, typename ... Others>
inline auto exterior_derivative(TF const & f, Others && ...others)
DECL_RET_TYPE((Field<Expression<ExteriorDerivative, TF>>(f,
						std::forward<Others>(others)...)))

template<typename TF, typename ... Others>
inline auto codifferential_derivative(TF const & f, Others && ...others)
DECL_RET_TYPE((Field<Expression<CodifferentialDerivative, TF>>(f,
						std::forward<Others>(others)...)))

template<typename ...T>
inline auto operator*(Field<T...> const & f)
DECL_RET_TYPE((hodge_star(f)))
;
template<typename ... T>
inline auto d(Field<T...> const & f)
DECL_RET_TYPE( (exterior_derivative(f)) )
;

template<typename ... T>
inline auto delta(Field<T...> const & f)
DECL_RET_TYPE( (codifferential_derivative(f)) )
;

template<unsigned int NDIMS, typename TL, typename ...T>
inline auto iv(nTuple<NDIMS, TL> const & v, Field<T...> const & f)
DECL_RET_TYPE( (interior_product(v,f)) )
;

template<typename ...T1, typename ... T2>
inline auto operator^(Field<T1...> const & lhs, Field<T2...> const & rhs)
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
