/**
 * @file calculus.h
 *
 *  Created on: 2014-9-23
 *      Author: salmon
 */

#ifndef CALCULUS_H_
#define CALCULUS_H_

#include <cstddef>
#include <type_traits>

#include "../gtl/expression_template.h"
#include "../gtl/macro.h"
#include "../gtl/mpl.h"
#include "../gtl/type_traits.h"
#include "../mesh/domain_traits.h"
#include "../mesh/mesh_ids.h"
#include "../mesh/mesh_traits.h"

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

/**
 *  @addtogroup diff_geo Differential Geometry
 *  @brief Differential geometry is a mathematical discipline that
 *  uses the techniques of @ref diff_calculus,@ref integral_calculus,
 *  @ref linear_algebra and @ref multilinear_algebra to study problems in geometry.
 *  @details Differential geometry is a mathematical discipline that
 *  uses the techniques of differential calculus, integral calculus,
 *  linear algebra and multilinear algebra to study problems in geometry.
 *   The theory of plane and space curves and surfaces in the three-dimensional
 *    Euclidean space formed the basis for development of differential
 *     geometry during the 18th century and the 19th century.
 */
/** @ingroup diff_geo
 *  @addtogroup diff_form Differential Form
 *  @{
 *  @brief In the mathematical fields of @ref diff_geo and tensor calculus,
 *   differential forms are an approach to multivariable calculus that
 *     is independent of coordinates. --wiki
 *
 *
 * ## Summary
 * \note Let \f$M\f$ be a _smooth manifold_. A _differential form_ of degree \f$k\f$ is
 *  a smooth section of the \f$k\f$th exterior power of the cotangent bundle of \f$M\f$.
 *  At any point \f$p \in M\f$, a k-form \f$\beta\f$ defines an alternating multilinear map
 * \f[
 *   \beta_p\colon T_p M\times \cdots \times T_p M \to \mathbb{R}
 * \f]
 * (with k factors of \f$T_p M\f$ in the product), where TpM is the tangent space to \f$M\f$ at \f$p\f$.
 *  Equivalently, \f$\beta\f$ is a totally antisymetric covariant tensor field of rank \f$k\f$.
 *
 *  Differential form is a field
 *
 * ## Requirements
 *
 *  @}
 *
 */

/**
 * @ingroup diff_geo
 * \addtogroup  manifold Manifold
 *  @{
 *    \brief   Discrete spatial-temporal space
 *
 * ## Summary
 *  Manifold
 * ## Requirements
 *
 ~~~~~~~~~~~~~{.cpp}
 template <typename Geometry,template<typename> class Policy1,template<typename> class Policy2>
 class Mesh:
 public Geometry,
 public Policy1<Geometry>,
 public Policy2<Geometry>
 {
 .....
 };
 ~~~~~~~~~~~~~
 * The following table lists requirements for a Mesh type `M`,
 *
 *  Pseudo-Signature  		| Semantics
 *  ------------------------|-------------
 *  `M( const M& )` 		| Copy constructor.
 *  `~M()` 				    | Destructor.
 *  `geometry_type`		    | Geometry type of manifold, which describes coordinates and metric
 *  `topology_type`		    | Topology structure of manifold,   Topology of grid points
 *  `coordiantes_type` 	    | data type of coordinates, i.e. nTuple<3,Real>
 *  `index_type`			| data type of the index of grid points, i.e. unsigned long
 *  `Domain  domain()`	    | Root domain of manifold
 *
 *
 * Mesh policy concept {#concept_manifold_policy}
 * ================================================
 *   Poilcies define the behavior of manifold , such as  interpolator or calculus;
 ~~~~~~~~~~~~~{.cpp}
 template <typename Geometry > class P;
 ~~~~~~~~~~~~~
 *
 *  The following table lists requirements for a Mesh policy type `P`,
 *
 *  Pseudo-Signature  	   | Semantics
 *  -----------------------|-------------
 *  `P( Geometry  & )` 	   | Constructor.
 *  `P( P const  & )`	   | Copy constructor.
 *  `~P( )` 			   | Copy Destructor.
 *
 * ## Interpolator policy
 *   Interpolator, map between discrete space and continue space, i.e. Gather & Scatter
 *
 *    Pseudo-Signature  	   | Semantics
 *  ---------------------------|-------------
 *  `gather(field_type const &f, coordinate_tuple x  )` 	    | gather data from `f` at coordinates `x`.
 *  `scatter(field_type &f, coordinate_tuple x ,value_type v)` 	| scatter `v` to field  `f` at coordinates `x`.
 *
 * ## Calculus  policy
 *  Define calculus operation of  fields on the manifold, such  as algebra or differential calculus.
 *  Differential calculus scheme , i.e. FDM,FVM,FEM,DG ....
 *
 *
 *  Pseudo-Signature  		| Semantics
 *  ------------------------|-------------
 *  `calculate(TOP op, field_type const &f, field_type const &f, index_type s ) `	| `calculate`  binary operation `op` at grid point `s`.
 *  `calculate(TOP op, field_type const &f,  index_type s )` 	| `calculate`  unary operation  `op`  at grid point `s`.
 *
 *  *
 *  @}
 */
namespace tags
{
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

struct MapTo;
}  // namespace tags

namespace traits
{
template<typename T>
struct iform<_Field<Expression<tags::HodgeStar, T> > > : public std::integral_constant<
		int, traits::rank<T>::value - traits::iform<T>::value>
{
};

template<typename T0, typename T1>
struct iform<_Field<Expression<tags::InteriorProduct, T0, T1> > > : public std::integral_constant<
		int, traits::iform<T1>::value - 1>
{

};

template<typename T>
struct iform<_Field<Expression<tags::ExteriorDerivative, T> > > : public std::integral_constant<
		int, traits::iform<T>::value + 1>
{
};

template<typename T>
struct iform<_Field<Expression<tags::CodifferentialDerivative, T> > > : public std::integral_constant<
		int, traits::iform<T>::value - 1>
{
};

template<typename T0, typename T1>
struct iform<_Field<Expression<tags::Wedge, T0, T1> > > : public std::integral_constant<
		int, iform<T0>::value + iform<T1>::value>
{
};

template<size_t I, typename T1>
struct iform<_Field<Expression<tags::MapTo,  //
		std::integral_constant<int, I>, T1> > > : public std::integral_constant<
		int, I>
{
};

template<typename T>
struct value_type<_Field<Expression<tags::ExteriorDerivative, T> > >
{
	typedef result_of_t<
			simpla::_impl::multiplies(scalar_type_t<mesh_type_t<T>>,
					value_type_t<T>)> type;
};

template<typename T>
struct value_type<_Field<Expression<tags::CodifferentialDerivative, T> > >
{
	typedef result_of_t<
			simpla::_impl::multiplies(scalar_type_t<mesh_type_t<T>>,
					value_type_t<T>)> type;
};

template<typename T0, typename T1>
struct value_type<_Field<Expression<tags::Wedge, T0, T1> > >
{

	typedef result_of_t<
			simpla::_impl::multiplies(value_type_t<T0>, value_type_t<T1>)> type;
};

template<typename T0, typename T1>
struct value_type<_Field<Expression<tags::InteriorProduct, T0, T1> > >
{

	typedef result_of_t<
			simpla::_impl::multiplies(value_type_t<T0>, value_type_t<T1>)> type;
};

template<typename T0, typename T1>
struct value_type<_Field<Expression<tags::MapTo, T0, T1> > >
{
	typedef value_type_t<T1> type;
};

namespace _impl
{
template<typename ...T> struct first_domain;

template<typename ...T> using first_domain_t=typename first_domain<T...>::type;

template<typename T0> struct first_domain<T0>
{
	typedef domain_t<T0> type;
};
template<typename T0, typename ...T> struct first_domain<T0, T...>
{
	typedef typename std::conditional<
			std::is_same<first_domain_t<T0>, std::nullptr_t>::value,
			typename first_domain<T...>::type, first_domain_t<T0> >::type type;
};

}  // namespace _impl

template<typename TAG, typename ...T>
struct domain_type<_Field<Expression<TAG, T...> > >
{

	typedef mpl::replace_tuple_t<1,
			typename iform<_Field<Expression<TAG, T...> > >::type,
			_impl::first_domain_t<T...> > type;

};

template<size_t I, typename T1>
struct domain_type<
		_Field<Expression<tags::MapTo, std::integral_constant<int, I>, T1> > >
{
	typedef mpl::replace_tuple_t<1, std::integral_constant<int, I>, domain_t<T1>> type;
};
}  // namespace traits
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

template<size_t I, typename T>
inline _Field<Expression<tags::MapTo, std::integral_constant<int, I>, T> > map_to(
		T const & f)
{
	return std::move(
			(_Field<Expression<tags::MapTo, std::integral_constant<int, I>, T> >(
					std::integral_constant<int, I>(), f)));
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

template<int I, typename T>
inline auto p_exterior_derivative(T const & f)
DECL_RET_TYPE(( _Field<Expression< tags:: ExteriorDerivative,
				std::integral_constant<int, I>, T > >(
						std::integral_constant<int, I>(),f)))
;
template<int I, typename T>
inline auto p_codifferential_derivative(T const & f)
DECL_RET_TYPE((_Field< Expression<tags:: CodifferentialDerivative ,
				std::integral_constant<int, I>, T > >(
						std::integral_constant<int, I>(),f)))
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
