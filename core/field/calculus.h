/**
 * @file calculus.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef CALCULUS_H_
#define CALCULUS_H_

#include <cstdbool>
#include <cstddef>
#include <type_traits>

#include "../gtl/expression_template.h"
#include "../gtl/macro.h"
#include "../gtl/mpl.h"
#include "../gtl/type_traits.h"
#include "mesh_ids.h"

namespace simpla
{

template<typename ... > class _Field;
template<typename ... > class Expression;

/**
 * @ingroup diff_geo
 *  @{
 *   @addtogroup  linear_algebra Linear Algebra
 *    @brief  This module collects linear algebra operations between field/forms
 *
 *    \note  Linear algebra is the branch of mathematics concerning vector spaces and linear mappings between such spaces. --wiki
 *    \note \f$\Omega^n\f$ means fields/forms on N-dimensional manifold \f$M\f$.
 *  @}
 *
 */
/**
 *  @ingroup linear_algebra
 *  @addtogroup exterior_algebra  Exterior algebra on forms
 *  @{
 *
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{N-n}\f$ =HodgeStar(\f$\Omega^n\f$ )	| hodge star, abbr. operator *
 *  \f$\Omega^{m+n}\f$ =Wedge(\f$\Omega^m\f$ ,\f$\Omega^m\f$  )	| wedge product, abbr. operator^
 *  \f$\Omega^{n-1}\f$ =InteriorProduct(Vector Field ,\f$\Omega^n\f$  )	| interior product, abbr. iv
 *  \f$\Omega^{N}\f$ =InnerProduct(\f$\Omega^m\f$ ,\f$\Omega^m\f$ ) | inner product,
 *
 */
namespace tags
{
struct HodgeStar;
struct InteriorProduct;
struct Wedge;

struct ExteriorDerivative;
struct CodifferentialDerivative;

struct MapTo;
}  // namespace tags

template<typename ... T>
struct _Field<Expression<T...>> : public Expression<T...>
{
	using Expression<T...>::Expression;
};

namespace traits
{

template<typename ... T>
struct reference<_Field<Expression<T...> > >
{
	typedef _Field<Expression<T...> > type;
};

// namespace _impl
template<typename TAG, typename T0>
struct domain_type<_Field<Expression<TAG, T0> > >
{
	typedef typename domain_type<T0>::type type;
};
namespace _impl
{

template<typename ...> struct field_traits;

template<typename T>
struct field_traits<_Field<Expression<simpla::tags::HodgeStar, T> > >
{
private:
	static constexpr size_t NDIMS = traits::rank<T>::value;
//	static constexpr size_t IL = traits::iform<T>::value;
public:

	static const size_t ndims = NDIMS > -IL ? NDIMS : 0;
	static const size_t iform = NDIMS - IL;

	typedef traits::value_type_t<T> value_type;
	typedef traits::domain_t<T> domain_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;
};

template<typename TL, typename TR>
struct field_traits<_Field<Expression<simpla::tags::InteriorProduct, TL, TR> > >
{
private:
	static constexpr size_t NDIMS = mpl::max<size_t, traits::rank<TL>::value,
			traits::rank<TR>::value>::value;
//	static constexpr size_t IL = traits::iform<TL>::value;
//	static constexpr size_t IR = traits::iform<TR>::value;
//
	typedef traits::value_type_t<TL> l_type;
	typedef traits::value_type_t<TR> r_type;

public:
	static const size_t ndims = mpl::max<size_t, IL, IR>::value > 0 ? NDIMS : 0;
	static const size_t iform = mpl::max<size_t, IL, IR>::value - 1;

	typedef traits::result_of_t<simpla::_impl::multiplies(l_type, r_type)> value_type;
	typedef traits::domain_t<TL> domain_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;
};

template<typename TL, typename TR>
struct field_traits<_Field<Expression<simpla::tags::Wedge, TL, TR> > >
{
private:
	static constexpr size_t NDIMS = mpl::max<size_t, traits::rank<TL>::value,
			traits::rank<TR>::value>::value;
//	static constexpr size_t IL = traits::iform<TL>::value;
//	static constexpr size_t IR = traits::iform<TR>::value;

	typedef traits::value_type_t<TL> l_type;
	typedef traits::value_type_t<TR> r_type;
public:
	static const size_t ndims = IL + IR <= NDIMS ? NDIMS : 0;
	static const size_t iform = IL + IR;

	typedef traits::result_of_t<simpla::_impl::multiplies(l_type, r_type)> value_type;
	typedef traits::domain_t<TL> domain_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;
};

}
// namespace _impl
}// namespace traits
template<typename T>
inline auto hodge_star(T const & f)
DECL_RET_TYPE(( _Field<Expression<tags::HodgeStar , T > >(f)))

template<typename TL, typename TR>
inline auto wedge(TL const & l, TR const & r)
DECL_RET_TYPE((_Field< Expression<tags::Wedge , TL, TR> >(l, r)))

template<typename TL, typename TR>
inline auto interior_product(TL const & l, TR const & r)
DECL_RET_TYPE((_Field< Expression<tags::InteriorProduct , TL, TR > > (l, r)))

template<typename ...T>
inline auto operator*(_Field<T...> const & f)
DECL_RET_TYPE((hodge_star(f)))

template<size_t ndims, typename TL, typename ...T>
inline auto iv(nTuple<TL, ndims> const & v, _Field<T...> const & f)
DECL_RET_TYPE( (interior_product(v,f)) )

template<typename ...T1, typename ... T2>
inline auto operator^(_Field<T1...> const & lhs, _Field<T2...> const & rhs)
DECL_RET_TYPE( (wedge(lhs,rhs)) )
/** @} */
/**
 * @ingroup linear_algebra
 * @addtogroup  vector_algebra   Linear algebra on vector fields
 * @{
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$  	| negate operation
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$  	| positive operation
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$ +\f$\Omega^n\f$ 	| add
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$ -\f$\Omega^n\f$ 	| subtract
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$ *Scalar  	| multiply
 *  \f$\Omega^n\f$ = Scalar * \f$\Omega^n\f$  	| multiply
 *  \f$\Omega^n\f$ = \f$\Omega^n\f$ / Scalar  	| divide
 *
 */
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
		ENABLE_IF_DECL_RET_TYPE(( traits::iform<_Field<TL...>>::value==EDGE),
				wedge(lhs , rhs ))
;

template<typename ...TL, typename ...TR> inline auto cross(
		_Field<TL...> const & lhs, _Field<TR...> const & rhs)
		ENABLE_IF_DECL_RET_TYPE((traits::iform<_Field<TL...>>::value==FACE),
				hodge_star(wedge(hodge_star(lhs) , hodge_star(rhs) )))
;
template<typename TL, typename ... TR> inline auto dot(nTuple<TL, 3> const & v,
		_Field<TR...> const & f)
		DECL_RET_TYPE( (interior_product(v, f)))

template<typename ...TL, typename TR> inline auto dot(_Field<TL...> const & f,
		nTuple<TR, 3> const & v)
		DECL_RET_TYPE( (interior_product(v, f)))
;

template<typename ... TL, typename TR> inline auto cross(
		_Field<TL...> const & f, nTuple<TR, 3> const & v)
		DECL_RET_TYPE( (interior_product(v, hodge_star(f))))
;

template<typename ... T, typename TL> inline auto cross(_Field<T...> const & f,
		nTuple<TL, 3> const & v)
		DECL_RET_TYPE((interior_product(v, f)))
;
/** @} */
/**
 * @ingroup linear_algebra
 * @addtogroup linear_map Linear map between forms/fields.
 * @{
 *
 *   Map between vector form  and scalar form
 *
 *  Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{1}\f$ =MapTo(\f${V\Omega}^0\f$ )	| map vector 0-form to 1-form
 *  \f${V\Omega}^{0}\f$ =MapTo(\f$\Omega^1\f$ )	| map 1-form to vector 0-form
 *
 *  \f{eqnarray*}{
 *  R &=& 1-\sum_{s}\frac{\omega_{ps}^{2}}{\omega\left(\omega+\Omega_{s}\right)}\\
 *  L &=& 1+\sum_{s}\frac{\omega_{ps}^{2}}{\omega\left(\omega-\Omega_{s}\right)}\\
 *  P &=& 1-\sum_{s}\frac{\omega_{ps}^{2}}{\omega^{2}}
 *  \f}
 */

namespace traits
{
namespace _impl
{

template<size_t I, typename T>
struct field_traits<
		_Field<
				Expression<simpla::tags::MapTo,
						std::integral_constant<size_t, I>, T> > >
{
	static constexpr size_t NDIMS = traits::rank<T>::value;
public:
	static const size_t ndims = NDIMS;
	static const size_t iform = IR;

	typedef traits::value_type_t<T> value_type;
	typedef traits::domain_t<T> domain_type;
};

}
// namespace _impl
}// namespace traits

template<size_t I, typename T>
inline _Field<Expression<tags::MapTo, std::integral_constant<size_t, I>, T> > map_to(
		T const & f)
{
	return std::move(
			(_Field<Expression<tags::MapTo, std::integral_constant<size_t, T> > >(
					f)));
}

/** @} */
/**
 * @ingroup diff_geo
 * @{
 * @addtogroup  diff_calculus Differential calculus
 * @brief  This module collects differential calculus on fields or forms
 * @}
 */
/**
 * @ingroup diff_calculus
 * @addtogroup dif_calculus_form Differential calculus on forms
 * @{
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{n-1}\f$ =ExteriorDerivative(\f$\Omega^n\f$ )	| Exterior Derivative, abbr. d
 *  \f$\Omega^{n+1}\f$ =Codifferential(\f$\Omega^n\f$ )	| Codifferential Derivative, abbr. delta
 *
 */

namespace traits
{

namespace _impl
{

template<typename T>
struct field_traits<_Field<Expression<simpla::tags::ExteriorDerivative, T> > >
{
private:
	static constexpr size_t NDIMS = traits::rank<T>::value;
//	static constexpr size_t IL = traits::iform<T>::value;

public:
	static constexpr size_t ndims = IL < NDIMS ? NDIMS : 0;
	static constexpr size_t iform = IL + 1;
	static constexpr bool is_field = traits::is_field<T>::value;

	typedef traits::value_type_t<T> value_type;
	typedef traits::domain_t<T> domain_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;

};
template<typename T>
struct field_traits<_Field<simpla::tags::CodifferentialDerivative, T> >
{
private:
	static constexpr size_t NDIMS = traits::rank<T>::value;
//	static constexpr size_t IL = traits::iform<T>::value;
public:
	static const size_t ndims = IL > 0 ? NDIMS : 0;
	static const size_t iform = IL - 1;
	typedef traits::value_type_t<T> value_type;
	typedef traits::domain_t<T> domain_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;
};

}
// namespace _impl

}// namespace traits
template<typename T>
inline auto exterior_derivative(T const & f)
DECL_RET_TYPE(( _Field< Expression<tags::ExteriorDerivative , T > >(f)))

template<typename T>
inline auto codifferential_derivative(T const & f)
DECL_RET_TYPE((_Field<Expression< tags::CodifferentialDerivative , T > >(f)))

template<typename ... T>
inline auto d(_Field<T...> const & f)
DECL_RET_TYPE( (exterior_derivative(f)) )

template<typename ... T>
inline auto delta(_Field<T...> const & f)
DECL_RET_TYPE( (codifferential_derivative(f)) )
/**@}*/

/**
 *  @ingroup diff_calculus
 *  @addtogroup vector_calculus Differential calculus on fields
 *  @{
 *
 *  Pseudo-Signature  			| Semantics
 * -----------------------------|--------------
 * \f$\Omega^{1}\f$=Grad(\f$\Omega^0\f$ )		| Grad
 * \f$\Omega^{0}\f$=Diverge(\f$\Omega^1\f$ )	| Diverge
 * \f$\Omega^{2}\f$=Curl(\f$\Omega^1\f$ )		| Curl
 * \f$\Omega^{1}\f$=Curl(\f$\Omega^2\f$ )		| Curl
 *
 *
 */
template<typename ... T>
inline auto grad(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((traits::iform<_Field<T...>>::value==VERTEX),
		(exterior_derivative(f)))
;
template<typename ... T>
inline auto grad(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((traits::iform<_Field<T...>>::value==VOLUME),
		((codifferential_derivative(-f))) )
;
template<typename ...T>
inline auto diverge(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((traits::iform<_Field<T...>>::value==FACE),
		(exterior_derivative(f)))
;
template<typename ...T>
inline auto diverge(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((traits::iform<_Field<T...>>::value==EDGE),
		(codifferential_derivative(-f)))
;
template<typename ... T>
inline auto curl(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((traits::iform<_Field<T...>>::value==EDGE),
		(exterior_derivative(f)))
;
template<typename ... T>
inline auto curl(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((traits::iform<_Field<T...>>::value==FACE),
		((codifferential_derivative(-f))) )
;

namespace traits
{

namespace _impl
{

template<size_t I, typename T>
struct field_traits<
		_Field<
				Expression<simpla::tags::ExteriorDerivative,
						std::integral_constant<size_t, I>, T> > >
{
private:
	static constexpr size_t NDIMS = traits::rank<T>::value;
//	static constexpr size_t IL = traits::iform<T>::value;

public:
	static constexpr size_t ndims = IR < NDIMS ? NDIMS : 0;
	static constexpr size_t iform = IR + 1;
	static constexpr bool is_field = traits::is_field<T>::value;

	typedef traits::value_type_t<T> value_type;
	typedef traits::domain_t<T> domain_type;

};
template<size_t I, typename T>
struct field_traits<
		_Field<
				Expression<simpla::tags::CodifferentialDerivative,
						std::integral_constant<size_t, I>, T> > >
{
private:
	static constexpr size_t NDIMS = traits::rank<T>::value;
//	static constexpr size_t IL = traits::iform<T>::value;
public:
	static const size_t ndims = IR > 0 ? NDIMS : 0;
	static const size_t iform = IR - 1;
	typedef traits::value_type_t<T> value_type;
	typedef traits::domain_t<T> domain_type;

};
}  // namespace _impl

}  // namespace traits
template<size_t I, typename T>
inline auto p_exterior_derivative(T const & f)
DECL_RET_TYPE(( _Field<Expression< tags:: ExteriorDerivative,
				std::integral_constant<size_t, I>, T > >(f)))
;
template<size_t I, typename T>
inline auto p_codifferential_derivative(T const & f)
DECL_RET_TYPE((_Field< Expression<tags:: CodifferentialDerivative ,
				std::integral_constant<size_t, I>, T > >(f)))
;
template<typename T> inline auto curl_pdx(T const & f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value==EDGE,
		(p_exterior_derivative<0>(f )))
;
template<typename T> inline auto curl_pdy(T const & f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value==EDGE,
		(p_exterior_derivative<1>(f )))
;
template<typename T> inline auto curl_pdz(T const & f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value==EDGE,
		(p_exterior_derivative<2>(f )))
;
template<typename T> inline auto curl_pdx(T const & f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value==FACE,
		(p_codifferential_derivative<0>(f )))
;
template<typename T> inline auto curl_pdy(T const & f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value==FACE,
		(p_codifferential_derivative<1>(f )))
;
template<typename T> inline auto curl_pdz(T const & f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value==FACE,
		(p_codifferential_derivative<2>(f )))
;
/** @} */
}
// namespace simpla

#endif /* CALCULUS_H_ */
