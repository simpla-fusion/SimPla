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
template<typename ... > class _Field;
template<typename ... > class Expression;
template<typename ... > class Domain;

template<typename T>
struct field_traits
{
	static constexpr unsigned int iform = VERTEX;
};

template<typename TManifold, unsigned int IFORM, typename ...Others>
struct field_traits<_Field<Domain<TManifold, IFORM>, Others...>>
{
	static constexpr unsigned int iform = IFORM;
};

/// \defgroup  ExteriorAlgebra Exterior algebra
/// @{
struct HodgeStar
{
	template<typename TL typename TI>
	constexpr auto operator()(TL && l, TI const&s) const
	DECL_RET_TYPE((get_value(std::forward<TL>(l),s)  ))
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

template<typename ... T>
inline _Field<Expression<HodgeStar, _Field<T...>>> hodge_star(_Field<T...> const & f)
{
	return std::move((_Field<Expression<HodgeStar, _Field<T...>>>(f)));
}

template<typename ... T, typename TR>
inline _Field<Expression<Wedge, _Field<T...>, TR>> wedge(_Field<T...> const & l,
		TR const & r)
{
	return std::move(_Field<Expression<Wedge, _Field<T...>, TR>>(l, r));
}

template<typename ...T, typename TR>
inline auto interior_product(_Field<T...> const & l, TR const & r)
DECL_RET_TYPE ((_Field<Expression<InteriorProduct,_Field<T...>, TR>>(l, r)))

template<typename ...T, typename ... Others>
inline auto exterior_derivative(_Field<T...> const & f, Others && ...others)
DECL_RET_TYPE((_Field<Expression<ExteriorDerivative, _Field<T...>>>(f,
						std::forward<Others>(others)...)))

template<typename ...T, typename ... Others>
inline auto codifferential_derivative(_Field<T...> const & f,
		Others && ...others)
				DECL_RET_TYPE((_Field<Expression<CodifferentialDerivative, _Field<T...>>>(f,
										std::forward<Others>(others)...)))

template<typename ...T>
inline auto operator*(_Field<T...> const & f)
DECL_RET_TYPE((hodge_star(f)))
;
template<typename ... T>
inline auto d(_Field<T...> const & f)
DECL_RET_TYPE( (exterior_derivative(f)) )
;

template<typename ... T>
inline auto delta(_Field<T...> const & f)
DECL_RET_TYPE( (codifferential_derivative(f)) )
;

template<unsigned int NDIMS, typename TL, typename ...T>
inline auto iv(nTuple<NDIMS, TL> const & v, _Field<T...> const & f)
DECL_RET_TYPE( (interior_product(v,f)) )
;

template<typename ...T1, typename ... T2>
inline auto operator^(_Field<T1...> const & lhs, _Field<T2...> const & rhs)
DECL_RET_TYPE( (wedge(lhs,rhs)) )
;

///  @}

///  \defgroup  VectorAlgebra Vector algebra
///  @{
template<typename ...TL, typename TR> inline auto inner_product(
		_Field<TL...> const & lhs, TR const & rhs)
		DECL_RET_TYPE(wedge (lhs,hodge_star( rhs) ))
;

template<typename ...TL, typename TR> inline auto dot(_Field<TL...> const & lhs,
		TR const & rhs)
		DECL_RET_TYPE(wedge(lhs , hodge_star(rhs) ))
;
template<typename ...TL, typename ...TR> inline auto cross(
		_Field<TL...> const & lhs, _Field<TR...> const & rhs)
		ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<TL...>>::iform==EDGE),
				wedge(lhs , rhs ))
;

template<typename ...TL, typename ...TR> inline auto cross(
		_Field<TL...> const & lhs, _Field<TR...> const & rhs)
		ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<TL...>>::iform==FACE),
				hodge_star(wedge(hodge_star(lhs) , hodge_star(rhs) )))

template<typename TL, typename ... TR> inline auto dot(nTuple<3, TL> const & v,
		_Field<TR...> const & f)
		DECL_RET_TYPE( (interior_product(v, f)))
;

template<typename ...TL, typename TR> inline auto dot(_Field<TL...> const & f,
		nTuple<3, TR> const & v)
		DECL_RET_TYPE( (interior_product(v, f)))
;

template<typename ... TL, typename TR> inline auto cross(
		_Field<TL...> const & f, nTuple<3, TR> const & v)
		DECL_RET_TYPE( (interior_product(v, hodge_star(f))))
;

template<typename ... T, typename TL> inline auto cross(_Field<T...> const & f,
		nTuple<3, TL> const & v)
		DECL_RET_TYPE((interior_product(v, f)))
;

template<typename ...T>
inline auto grad(_Field<T...> const & f)
DECL_RET_TYPE( ( exterior_derivative(f)))
;

template<typename ... T>
inline auto grad(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==VERTEX),
		(exterior_derivative(f)))

template<typename ... T>
inline auto grad(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==FACE),
		(-(codifferential_derivative(f))) )

template<typename ...T>
inline auto diverge(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==FACE),
		(exterior_derivative(f)))
;

template<typename ...T>
inline auto diverge(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==EDGE),
		-(codifferential_derivative(f)))
;

template<typename ... T>
inline auto curl(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==EDGE),
		(exterior_derivative(f)))

template<typename ... T>
inline auto curl(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==FACE),
		(-(codifferential_derivative(f))) )
;

///   @}

///  \ingroup  FETL
///  \defgroup  NonstandardOperations Non-standard operations
///   @{
template<typename TM, typename TR> inline auto CurlPDX(
		_Field<Domain<TM, EDGE>, TR> const & f)
				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<unsigned int ,0>())))
;

template<typename TM, typename TR> inline auto CurlPDY(
		_Field<Domain<TM, EDGE>, TR> const & f)
				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<unsigned int ,1>())))
;

template<typename TM, typename TR> inline auto CurlPDZ(
		_Field<Domain<TM, EDGE>, TR> const & f)
				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<unsigned int ,2>())))
;

template<typename TM, typename TR> inline auto CurlPDX(
		_Field<Domain<TM, FACE>, TR> const & f)
				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<unsigned int ,0>())))
;

template<typename ...T> inline auto CurlPDY(
		_Field<T...> const & f)
				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<unsigned int ,1>())))
;

template<typename TM, typename TR>
inline auto CurlPDZ(
		_Field<Domain<TM, FACE>, TR> const & f)
				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<unsigned int ,2>())))
;

//template<unsigned int IL, typename TM, unsigned int IR, typename TR>
//inline auto MapTo(
//		_Field<Domain<TM, IR>, TR> const & f)
//				DECL_RET_TYPE( (_Field<Domain<TM,IL>, _Field<MAPTO,std::integral_constant<unsigned int ,IL>,_Field<Domain<TM, IR>, TR> > >(std::integral_constant<unsigned int ,IL>(), f)))
//;
//
//template<unsigned int IL, typename TM, unsigned int IR, typename TR>
//inline auto MapTo(std::integral_constant<unsigned int, IL>,
//		_Field<Domain<TM, IR>, TR> const & f)
//				DECL_RET_TYPE( (_Field<Domain<TM,IL>, _Field<MAPTO,std::integral_constant<unsigned int ,IL>,_Field<Domain<TM, IR>, TR> > >(std::integral_constant<unsigned int ,IL>(), f)))
//;

///   @}

}
// namespace simpla

#endif /* CALCULUS_H_ */
