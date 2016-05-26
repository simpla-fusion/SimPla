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
#include "../gtl/nTuple.h"
#include "../gtl/macro.h"
#include "../gtl/mpl.h"
#include "../gtl/type_traits.h"
#include "../mesh/MeshEntity.h"
#include "../field/Field.h"
#include "../field/FieldExpression.h"
#include "ManifoldTraits.h"

namespace simpla
{

template<typename ...> class Field;

template<typename ...> class Expression;

namespace traits
{


template<typename T> struct is_primary_complex { static constexpr bool value = false; };
template<typename T> struct is_primary_complex<std::complex<T>>
{
    static constexpr bool value = std::is_arithmetic<T>::value;
};
template<typename T> using is_primary_complex_t=  std::enable_if_t<is_primary_complex<T>::value>;

template<typename T> struct is_primary_scalar
{
    static constexpr bool value = std::is_arithmetic<T>::value || is_primary_complex<T>::value;
};
template<typename T> using is_primary_scalar_t=  std::enable_if_t<is_primary_scalar<T>::value>;


template<typename T> struct is_primary
{
    static constexpr bool value = (is_primary_scalar<T>::value || is_ntuple<T>::value) && !is_expression<T>::value;
};
template<typename T> using is_primary_t=  std::enable_if_t<is_primary<T>::value>;


//template<typename T> struct is_ntuple { static constexpr bool value = false; };
//template<typename T, int ...N> struct is_ntuple<nTuple<T, N...>> { static constexpr bool value = true; };
template<typename T> using is_primary_ntuple_t=std::enable_if_t<is_ntuple<T>::value && !(is_expression<T>::value)>;
template<typename T> using is_expression_ntuple_t=std::enable_if_t<is_ntuple<T>::value && (is_expression<T>::value)>;


template<typename T> struct is_expression { static constexpr bool value = false; };
template<typename ...T, template<typename ...> class F>
struct is_expression<F<Expression<T...>>> { static constexpr bool value = true; };
template<typename T> using is_expression_t=  std::enable_if_t<is_expression<T>::value>;



template<typename T> using is_field_t=  std::enable_if_t<is_field<T>::value>;
template<typename T> using is_primary_field_t=   std::enable_if_t<is_field<T>::value && !(is_expression<T>::value)>;
template<typename T> using is_expression_field_t=  std::enable_if_t<is_field<T>::value && (is_expression<T>::value)>;

}
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


using namespace simpla::mesh;

namespace traits
{

template<typename> class rank;

template<typename> class iform;

template<typename> class value_type;

template<typename ...T>
struct rank<Field<T...> > : public std::integral_constant<int, 3> { };


template<typename TV, typename TM, typename TFORM, typename ...Others>
struct iform<Field<TV, TM, TFORM, Others...> > : public TFORM { };


template<typename T>
struct iform<Field<Expression<calculus::tags::HodgeStar, T> > > : public std::integral_constant<
        int, rank<T>::value - iform<T>::value>
{
};

template<typename T0, typename T1>
struct iform<Field<Expression<calculus::tags::InteriorProduct, T0, T1> > > : public std::integral_constant<
        int, iform<T1>::value - 1>
{
};

template<typename T>
struct iform<Field<Expression<calculus::tags::ExteriorDerivative, T> > > : public std::integral_constant<
        int, iform<T>::value + 1>
{
};

template<typename T>
struct iform<Field<Expression<calculus::tags::CodifferentialDerivative, T> > > : public std::integral_constant<
        int, iform<T>::value - 1>
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

template<size_t I, typename T1>
struct iform<Field<Expression<calculus::tags::MapTo, T1, std::integral_constant<int, I>>> >
        : public std::integral_constant<int, I>
{
};

template<typename TV, typename ...Other>
struct value_type<Field<TV, Other...>> { typedef TV type; };

template<typename T>
struct value_type<Field<Expression<calculus::tags::HodgeStar, T> > > { typedef value_type_t <T> type; };


template<typename T>
struct value_type<Field<Expression<calculus::tags::ExteriorDerivative, T> > >
{
    typedef traits::result_of_t<simpla::_impl::multiplies(
            geometry::traits::scalar_type_t<traits::manifold_type_t<T >>, value_type_t <T>)> type;
};

template<typename T>
struct value_type<Field<Expression<calculus::tags::CodifferentialDerivative, T> > >
{
    typedef result_of_t<
            simpla::_impl::multiplies(geometry::traits::scalar_type_t<traits::manifold_type_t<T >>,
                                      value_type_t <T>)> type;
};

template<typename T0, typename T1>
struct value_type<Field<Expression<calculus::tags::Wedge, T0, T1> > >
{

    typedef traits::result_of_t<
            simpla::_impl::multiplies(value_type_t < T0 > , value_type_t < T1 > )> type;
};

template<typename T0, typename T1>
struct value_type<Field<Expression<calculus::tags::InteriorProduct, T0, T1> > >
{

    typedef traits::result_of_t<
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


template<typename T, typename ...Others>
struct value_type<Field<Expression<calculus::tags::MapTo, T, Others...> > > { typedef value_type_t <T> type; };
template<typename T>
struct value_type<Field<Expression<calculus::tags::MapTo, T, std::integral_constant<int, VERTEX>>> >
{
    typedef typename std::conditional_t<
            iform<T>::value == VERTEX || iform<T>::value == mesh::VOLUME,
            value_type_t < T>, simpla::nTuple<value_type_t < T>, 3> > type;
};

template<typename T>
struct value_type<Field<Expression<calculus::tags::MapTo, T, std::integral_constant<int, mesh::VOLUME>>> >
{
    typedef typename std::conditional_t<
            iform<T>::value == VERTEX || iform<T>::value == mesh::VOLUME,
            value_type_t < T>, simpla::nTuple<value_type_t < T>, 3> > type;
};


//template<typename TV, typename TM, typename ...Others>
//struct value_type<Field<Expression<calculus::tags::MapTo, Field<
//        nTuple < TV, 3>, TM, std::integral_constant<int, VERTEX>, Others...>,
//        std::integral_constant<int, mesh::EDGE> > > >
//{
//typedef TV type;
//};
//template<typename TV, typename TM, typename ...Others>
//struct value_type<Field<Expression<calculus::tags::MapTo, Field<
//        nTuple < TV, 3>, TM, std::integral_constant<int, VERTEX>, Others...>,
//        std::integral_constant<int, mesh::FACE> > > >
//{
//typedef TV type;
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
 *  \f$\Omega^{m+n}\f$ =wedge(\f$\Omega^m\f$ ,\f$\Omega^m\f$  )	| wedge product, abbr. operator^
 *  \f$\Omega^{n-1}\f$ =InteriorProduct(Vector field ,\f$\Omega^n\f$  )	| interior product, abbr. iv
 *  \f$\Omega^{N}\f$ =InnerProduct(\f$\Omega^m\f$ ,\f$\Omega^m\f$ ) | inner product,

 **/
template<typename T>
inline auto hodge_star(T const &f) DECL_RET_TYPE((Field<Expression<calculus::tags::HodgeStar, T> >(f)))

template<typename TL, typename TR>
inline auto wedge(TL const &l, TR const &r) DECL_RET_TYPE((Field<Expression<calculus::tags::Wedge, TL, TR> >(l, r)))

template<typename TL, typename TR>
inline auto interior_product(TL const &l, TR const &r) DECL_RET_TYPE(
        (Field<Expression<calculus::tags::InteriorProduct, TL, TR> >(l, r)))

template<typename ...T>
inline auto operator*(Field<T...> const &f) DECL_RET_TYPE((hodge_star(f)))

template<size_t ndims, typename TL, typename ...T>
inline auto iv(nTuple <TL, ndims> const &v, Field<T...> const &f) DECL_RET_TYPE((interior_product(v, f)))

template<typename ...T1, typename ... T2>
inline auto operator^(Field<T1...> const &lhs, Field<T2...> const &rhs) DECL_RET_TYPE((wedge(lhs, rhs)))

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
template<typename ...TL, typename TR>
inline auto inner_product(Field<TL...> const &lhs, TR const &rhs) DECL_RET_TYPE(wedge(lhs, hodge_star(rhs)));

template<typename ...TL, typename TR>
inline auto dot(Field<TL...> const &lhs, TR const &rhs) DECL_RET_TYPE(wedge(lhs, hodge_star(rhs)));


template<typename TL, typename TR>
inline auto wedge(TL const &l, TR const &r)
ENABLE_IF_DECL_RET_TYPE(traits::iform<TL>::value == mesh::VERTEX,
                        (Field<Expression<calculus::tags::Cross, TL, TR> >(l, r)))

template<typename ...TL, typename ...TR> inline auto cross(Field<TL...> const &lhs, Field<TR...> const &rhs)
ENABLE_IF_DECL_RET_TYPE((traits::iform<Field<TL... >>::value == mesh::EDGE), wedge(lhs, rhs));

template<typename ...TL, typename ...TR> inline auto cross(Field<TL...> const &lhs, Field<TR...> const &rhs)
ENABLE_IF_DECL_RET_TYPE((traits::iform<Field<TL... >>::value == mesh::FACE),
                        hodge_star(wedge(hodge_star(lhs), hodge_star(rhs))));


template<typename ...TL, typename ...TR> inline auto cross(Field<TL...> const &lhs, Field<TR...> const &rhs)
ENABLE_IF_DECL_RET_TYPE((traits::iform<Field<TL... >>::value == mesh::VERTEX),
                        (Field<Expression<calculus::tags::Cross, Field<TL...>, Field<TR...> > >(lhs, rhs)));

template<typename ...TL, typename ...TR> inline auto dot(Field<TL...> const &lhs, Field<TR...> const &rhs)
ENABLE_IF_DECL_RET_TYPE((traits::iform<Field<TL... >>::value == mesh::VERTEX),
                        (Field<Expression<calculus::tags::Dot, Field<TL...>, Field<TR...> > >(lhs, rhs)));


template<typename TL, typename ... TR> inline auto
dot(nTuple<TL, 3> const &v, Field<TR...> const &f) DECL_RET_TYPE((interior_product(v, f)))

template<typename ...TL, typename TR> inline auto
dot(Field<TL...> const &f, nTuple<TR, 3> const &v) DECL_RET_TYPE((interior_product(v, f)));

template<typename ... TL, typename TR> inline auto
cross(nTuple<TR, 3> const &v, Field<TL...> const &f) DECL_RET_TYPE((interior_product(v, hodge_star(f))));

template<typename ... T, typename TL> inline auto
cross(Field<T...> const &f, nTuple<TL, 3> const &v) DECL_RET_TYPE((interior_product(v, f)));

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

template<int I, typename T> inline auto
map_to(T const &f)
{
    return std::move((Field<Expression<calculus::tags::MapTo, T,
            std::integral_constant<int, I>>>(f, std::integral_constant<int, I>())));
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

template<typename T> inline auto
exterior_derivative(T const &f) DECL_RET_TYPE((Field<Expression<calculus::tags::ExteriorDerivative, T> >(f)))

template<typename T> inline auto
codifferential_derivative(T const &f) DECL_RET_TYPE(
        (Field<Expression<calculus::tags::CodifferentialDerivative, T> >(f)))
//
//template<typename ... T> inline auto
//d(Field<T...> const &f) DECL_RET_TYPE((exterior_derivative(f)))
//
//template<typename ... T> inline auto
//delta(Field<T...> const &f) DECL_RET_TYPE((codifferential_derivative(f)))
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
template<typename T> inline auto
grad(T const &f, I_const <mesh::VERTEX>) DECL_RET_TYPE((exterior_derivative(f)));

template<typename T> inline auto
grad(T const &f, I_const <mesh::VOLUME>) DECL_RET_TYPE(((codifferential_derivative(-f))));

template<typename T> inline auto
grad(T const &f) DECL_RET_TYPE((grad(f, traits::iform<T>())));

template<typename T> inline auto
diverge(T const &f, I_const <mesh::EDGE>) DECL_RET_TYPE((exterior_derivative(f)));

template<typename T> inline auto
diverge(T const &f, I_const <mesh::FACE>) DECL_RET_TYPE((codifferential_derivative(-f)));

template<typename T> inline auto
diverge(T const &f) DECL_RET_TYPE((diverge(f, traits::iform<T>())));


template<typename T> inline auto
curl(T const &f, I_const <mesh::EDGE>) { return exterior_derivative(f); }

template<typename T> inline auto
curl(T const &f, I_const <mesh::FACE>) { return codifferential_derivative(-f); }

template<typename T> inline auto
curl(T const &f) { return curl(f, traits::iform<T>()); }

template<int I, typename T> inline auto
p_exterior_derivative(T const &f) DECL_RET_TYPE((Field<Expression<calculus::tags::ExteriorDerivative,
        std::integral_constant<int, I>, T> >(std::integral_constant<int, I>(), f)));

template<int I, typename T> inline auto
p_codifferential_derivative(T const &f) DECL_RET_TYPE(
        (Field<Expression<calculus::tags::CodifferentialDerivative,
                std::integral_constant<int, I>, T> >(std::integral_constant<int, I>(), f)));

template<typename T> inline auto
curl_pdx(T const &f, I_const <mesh::EDGE>) DECL_RET_TYPE((p_exterior_derivative<0>(f)));

template<typename T> inline auto
curl_pdx(T const &f, I_const <mesh::FACE>) DECL_RET_TYPE((p_codifferential_derivative<0>(f)));

template<typename T> inline auto
curl_pdx(T const &f) { return curl_pdx(f, traits::iform<T>()); }

template<typename T> inline auto
curl_pdy(T const &f, I_const <mesh::EDGE>) DECL_RET_TYPE((p_exterior_derivative<1>(f)));

template<typename T> inline auto
curl_pdy(T const &f, I_const <mesh::FACE>) DECL_RET_TYPE((p_codifferential_derivative<1>(f)));

template<typename T> inline auto
curl_pdy(T const &f) { return curl_pdy(f, traits::iform<T>()); }

template<typename T> inline auto
curl_pdz(T const &f, I_const <mesh::EDGE>) DECL_RET_TYPE((p_exterior_derivative<2>(f)));

template<typename T> inline auto
curl_pdz(T const &f, I_const <mesh::FACE>) DECL_RET_TYPE((p_codifferential_derivative<2>(f)));

template<typename T> inline auto
curl_pdz(T const &f) { return curl_pdz(f, traits::iform<T>()); }
/** @} */

/** @} */


}
// namespace simpla

#endif /* CALCULUS_H_ */
