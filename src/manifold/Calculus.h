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

#include "../gtl/ExpressionTemplate.h"
#include "../gtl/macro.h"
#include "../gtl/mpl.h"
#include "../gtl/type_traits.h"
#include "../mesh/MeshEntity.h"
#include "ManifoldTraits.h"

namespace simpla
{

template<typename ...> class Field;

template<typename ...> class Expression;

/**
 * @ingroup diff_geo
 * @defgroup calculus Calculus on Manifold
 * @ingroup calculus
 * @{
 **/

namespace calculus { namespace tags
{
struct HodgeStar { };
struct InteriorProduct { };
struct Wedge { };

struct ExteriorDerivative { };
struct CodifferentialDerivative { };

struct MapTo { };

struct Cross { };

struct Dot { };
}}//namespace calculus// // namespace tags



namespace traits
{
using namespace simpla::mesh;

template<typename ...T>
struct rank<Field<T...> > : public std::integral_constant<int, 3>
{
};

template<typename T>
struct iform<Field<Expression<calculus::tags::HodgeStar, T> > > : public std::integral_constant<
        int, traits::rank<T>::value - traits::iform<T>::value>
{
};

template<typename T0, typename T1>
struct iform<Field<Expression<calculus::tags::InteriorProduct, T0, T1> > > : public std::integral_constant<
        int, traits::iform<T1>::value - 1>
{
};

template<typename T>
struct iform<Field<Expression<calculus::tags::ExteriorDerivative, T> > > : public std::integral_constant<
        int, traits::iform<T>::value + 1>
{
};

template<typename T>
struct iform<Field<Expression<calculus::tags::CodifferentialDerivative, T> > > : public std::integral_constant<
        int, traits::iform<T>::value - 1>
{
};
template<typename T0, typename T1>
struct iform<Field<Expression<calculus::tags::Wedge, T0, T1> > > : public std::integral_constant<
        int, iform<T0>::value + iform<T1>::value>
{
};
template<int I>
struct iform<std::integral_constant<int, I> > : public std::integral_constant<int, I>
{
};

template<typename T>
struct value_type<Field<Expression<calculus::tags::HodgeStar, T> > >
{
    typedef value_type_t <T> type;
};


template<typename T>
struct value_type<Field<Expression<calculus::tags::ExteriorDerivative, T> > >
{
    typedef result_of_t<simpla::_impl::multiplies(
            geometry::traits::scalar_type_t<manifold_type_t < T >> , value_type_t <T> )> type;
};

template<typename T>
struct value_type<Field<Expression<calculus::tags::CodifferentialDerivative, T> > >
{
    typedef result_of_t<
            simpla::_impl::multiplies(geometry::traits::scalar_type_t<manifold_type_t < T >> ,
    value_type_t <T> )> type;
};

template<typename T0, typename T1>
struct value_type<Field<Expression<calculus::tags::Wedge, T0, T1> > >
{

    typedef result_of_t<
            simpla::_impl::multiplies(value_type_t < T0 > , value_type_t < T1 > )> type;
};

template<typename T0, typename T1>
struct value_type<Field<Expression<calculus::tags::InteriorProduct, T0, T1> > >
{

    typedef result_of_t<
            simpla::_impl::multiplies(value_type_t < T0 > , value_type_t < T1 > )> type;
};


template<typename T0, typename T1>
struct value_type<Field<Expression<calculus::tags::Cross, T0, T1> > >
{

    typedef value_type_t <T0> type;
};

template<typename T0, typename T1>
struct value_type<Field<Expression<calculus::tags::Dot, T0, T1> > >
{

    typedef value_type_t <value_type_t<T0>> type;
};


template<size_t I, typename T1>
struct iform<Field<Expression<calculus::tags::MapTo, T1, std::integral_constant<int, I>>> >
        : public std::integral_constant<int, I>
{
};


template<typename T, typename ...Others>
struct value_type<Field<Expression<calculus::tags::MapTo, T, Others...> > >
{
    typedef value_type_t <T> type;
};
template<typename T>
struct value_type<Field<Expression<calculus::tags::MapTo, T, std::integral_constant<int, VERTEX>>> >
{
    typedef typename std::conditional<
            iform<T>::value == VERTEX || iform<T>::value == VOLUME,
            value_type_t < T>, nTuple<value_type_t<T>, 3> >::type type;
};


template<typename T>
struct value_type<Field<Expression<calculus::tags::MapTo, T, std::integral_constant<int, VOLUME>>> >
{
    typedef typename std::conditional<
            iform<T>::value == VERTEX || iform<T>::value == VOLUME,
            value_type_t < T>, nTuple<value_type_t<T>, 3> >::type type;
};


template<typename TV, typename TM, typename ...Others>
struct value_type<Field<Expression<calculus::tags::MapTo, Field<
        nTuple < TV, 3>, TM, std::integral_constant<int, VERTEX>, Others...>,
        std::integral_constant<int, EDGE> > > >
{
typedef TV type;
};

template<typename TV, typename TM, typename ...Others>
struct value_type<Field<Expression<calculus::tags::MapTo, Field<
        nTuple < TV, 3>, TM, std::integral_constant<int, VERTEX>, Others...>,
        std::integral_constant<int, FACE> > > >
{
typedef TV type;
};
//namespace _impl
//{
//template<typename ...T> struct first_domain;
//
//template<typename ...T> using first_domain_t=typename first_domain<T...>::type;
//
//template<typename T0> struct first_domain<T0>
//{
//    typedef domain_t<T0> type;
//};
//template<typename T0, typename ...T> struct first_domain<T0, T...>
//{
//    typedef typename std::conditional<
//            std::is_same<first_domain_t<T0>, std::nullptr_t>::value,
//            typename first_domain<T...>::type, first_domain_t<T0> >::type type;
//};
//
//}  // namespace _impl
//
//template<typename TAG, typename ...T>
//struct domain_type<field<Expression<TAG, T...> > >
//{
//
//    typedef mpl::replace_tuple_t<1,
//            typename iform<field<Expression<TAG, T...> > >::type,
//            _impl::first_domain_t<T...> > type;
//
//};
//
//template<size_t I, typename T1>
//struct domain_type<
//        field<Expression<calculus::tags::MapTo, std::integral_constant<int, I>, T1> > >
//{
//    typedef mpl::replace_tuple_t<1, std::integral_constant<int, I>, domain_t<T1>> type;
//};
}  // namespace traits



/**
 * @defgroup exterior_algebra Exterior algebra on forms
 * @{
 *
 *
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{N-n}\f$ =HodgeStar(\f$\Omega^n\f$ )	| hodge star, abbr. operator *
 *  \f$\Omega^{m+n}\f$ =Wedge(\f$\Omega^m\f$ ,\f$\Omega^m\f$  )	| wedge product, abbr. operator^
 *  \f$\Omega^{n-1}\f$ =InteriorProduct(Vector field ,\f$\Omega^n\f$  )	| interior product, abbr. iv
 *  \f$\Omega^{N}\f$ =InnerProduct(\f$\Omega^m\f$ ,\f$\Omega^m\f$ ) | inner product,

 **/
template<typename T>
inline auto hodge_star(T const &f)
DECL_RET_TYPE((Field < Expression < calculus::tags::HodgeStar, T > > (f)))

template<typename TL, typename TR>
inline auto wedge(TL const &l, TR const &r)
DECL_RET_TYPE((Field < Expression < calculus::tags::Wedge, TL, TR > > (l, r)))

template<typename TL, typename TR>
inline auto interior_product(TL const &l, TR const &r)
DECL_RET_TYPE((Field < Expression < calculus::tags::InteriorProduct, TL, TR > > (l, r)))

template<typename ...T>
inline auto operator*(Field<T...> const &f)
DECL_RET_TYPE((hodge_star(f)))

template<size_t ndims, typename TL, typename ...T>
inline auto iv(nTuple <TL, ndims> const &v, Field<T...> const &f)
DECL_RET_TYPE((interior_product(v, f)))

template<typename ...T1, typename ... T2>
inline auto operator^(Field<T1...> const &lhs, Field<T2...> const &rhs)
DECL_RET_TYPE((wedge(lhs, rhs)))

/** @} */

/**
 * @defgroup  vector_algebra   Linear algebra on vector fields
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
template<typename ...TL, typename TR> inline auto inner_product(Field<TL...> const &lhs, TR const &rhs)
DECL_RET_TYPE(wedge(lhs, hodge_star(rhs)));

template<typename ...TL, typename TR> inline auto dot(Field<TL...> const &lhs, TR const &rhs)
DECL_RET_TYPE(wedge(lhs, hodge_star(rhs)));


template<typename TL, typename TR>
inline auto wedge(TL const &l, TR const &r)
ENABLE_IF_DECL_RET_TYPE(traits::iform<TL>::value == VERTEX,
                        (Field < Expression < calculus::tags::Cross, TL, TR > > (l, r)))

template<typename ...TL, typename ...TR> inline auto cross(Field<TL...> const &lhs, Field<TR...> const &rhs)
ENABLE_IF_DECL_RET_TYPE((traits::iform < Field < TL... >>::value == EDGE), wedge(lhs, rhs));

template<typename ...TL, typename ...TR> inline auto cross(Field<TL...> const &lhs, Field<TR...> const &rhs)
ENABLE_IF_DECL_RET_TYPE((traits::iform < Field < TL... >>::value == FACE),
                        hodge_star(wedge(hodge_star(lhs), hodge_star(rhs))));


template<typename ...TL, typename ...TR> inline auto cross(Field<TL...> const &lhs, Field<TR...> const &rhs)
ENABLE_IF_DECL_RET_TYPE((traits::iform < Field < TL... >>::value == VERTEX),
                        (Field < Expression < calculus::tags::Cross, Field < TL...>, Field<TR...> > >(lhs, rhs)));

template<typename ...TL, typename ...TR> inline auto dot(Field<TL...> const &lhs, Field<TR...> const &rhs)
ENABLE_IF_DECL_RET_TYPE((traits::iform < Field < TL... >>::value == VERTEX),
                        (Field < Expression < calculus::tags::Dot, Field < TL...>, Field<TR...> > >(lhs, rhs)));


template<typename TL, typename ... TR> inline auto dot(nTuple<TL, 3> const &v, Field<TR...> const &f)
DECL_RET_TYPE((interior_product(v, f)))

template<typename ...TL, typename TR> inline auto dot(Field<TL...> const &f, nTuple<TR, 3> const &v)
DECL_RET_TYPE((interior_product(v, f)));

template<typename ... TL, typename TR> inline auto cross(Field<TL...> const &f, nTuple<TR, 3> const &v)
DECL_RET_TYPE((interior_product(v, hodge_star(f))));

template<typename ... T, typename TL> inline auto cross(Field<T...> const &f, nTuple<TL, 3> const &v)
DECL_RET_TYPE((interior_product(v, f)));

/** @} */
/**
 * @defgroup linear_map Linear map between forms/fields.
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

template<int I, typename T>
inline Field <Expression<calculus::tags::MapTo, T, std::integral_constant<int, I>>> map_to(
        T const &f)
{
    return std::move((Field < Expression < calculus::tags::MapTo, T,
            std::integral_constant<int, I>>>(f, std::integral_constant<int, I>())

    ));
}

/** @} */

/**
 * @defgroup dif_calculus_form Differential calculus on forms
 * @{
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{n-1}\f$ =ExteriorDerivative(\f$\Omega^n\f$ )	| Exterior Derivative, abbr. d
 *  \f$\Omega^{n+1}\f$ =Codifferential(\f$\Omega^n\f$ )	| Codifferential Derivative, abbr. delta
 *
 */

template<typename T>
inline auto exterior_derivative(T const &f)
DECL_RET_TYPE((Field < Expression < calculus::tags::ExteriorDerivative, T > > (f)))

template<typename T>
inline auto codifferential_derivative(T const &f)
DECL_RET_TYPE((Field < Expression < calculus::tags::CodifferentialDerivative, T > > (f)))

template<typename ... T>
inline auto d(Field<T...> const &f)
DECL_RET_TYPE((exterior_derivative(f)))

template<typename ... T>
inline auto delta(Field<T...> const &f)
DECL_RET_TYPE((codifferential_derivative(f)))
/**@}*/

/**
 *  @defgroup vector_calculus Differential calculus on fields
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
inline auto grad(Field<T...> const &f)
ENABLE_IF_DECL_RET_TYPE((traits::iform < Field < T... >>::value == VERTEX),
                        (exterior_derivative(f)));

template<typename ... T>
inline auto grad(Field<T...> const &f)
ENABLE_IF_DECL_RET_TYPE((traits::iform < Field < T... >>::value == VOLUME),
                        ((codifferential_derivative(-f))));

template<typename ...T>
inline auto diverge(Field<T...> const &f)
ENABLE_IF_DECL_RET_TYPE((traits::iform < Field < T... >>::value == FACE),
                        (exterior_derivative(f)));

template<typename ...T>
inline auto diverge(Field<T...> const &f)
ENABLE_IF_DECL_RET_TYPE((traits::iform < Field < T... >>::value == EDGE),
                        (codifferential_derivative(-f)));

template<typename ... T>
inline auto curl(Field<T...> const &f)
ENABLE_IF_DECL_RET_TYPE((traits::iform < Field < T... >>::value == EDGE),
                        (exterior_derivative(f)));

template<typename ... T>
inline auto curl(Field<T...> const &f)
ENABLE_IF_DECL_RET_TYPE((traits::iform < Field < T... >>::value == FACE),
                        ((codifferential_derivative(-f))));

template<int I, typename T>
inline auto p_exterior_derivative(T const &f)
DECL_RET_TYPE((Field < Expression < calculus::tags::ExteriorDerivative,
        std::integral_constant<int, I>, T > > (
        std::integral_constant<int, I>(), f)));

template<int I, typename T>
inline auto p_codifferential_derivative(T const &f)
DECL_RET_TYPE((Field < Expression < calculus::tags::CodifferentialDerivative,
        std::integral_constant<int, I>, T > > (
        std::integral_constant<int, I>(), f)));

template<typename T> inline auto curl_pdx(T const &f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value == EDGE,
                        (p_exterior_derivative<0>(f)));

template<typename T> inline auto curl_pdy(T const &f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value == EDGE,
                        (p_exterior_derivative<1>(f)));

template<typename T> inline auto curl_pdz(T const &f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value == EDGE,
                        (p_exterior_derivative<2>(f)));

template<typename T> inline auto curl_pdx(T const &f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value == FACE,
                        (p_codifferential_derivative<0>(f)));

template<typename T> inline auto curl_pdy(T const &f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value == FACE,
                        (p_codifferential_derivative<1>(f)));

template<typename T> inline auto curl_pdz(T const &f)
ENABLE_IF_DECL_RET_TYPE(traits::iform<T>::value == FACE,
                        (p_codifferential_derivative<2>(f)));
/** @} */

/** @} */


}
// namespace simpla

#endif /* CALCULUS_H_ */
