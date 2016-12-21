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
#include <simpla/toolbox/port_cxx14.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/toolbox/macro.h>

#include <simpla/toolbox/type_traits.h>
#include <simpla/mesh/MeshCommon.h>


namespace simpla
{

template<typename ...> class Expression;
/**
 * @ingroup diff_geo
 * @defgroup calculus Calculus on CoordinateSystem
 * @ingroup calculus
 * @{
 **/

namespace calculus { namespace tags
{
struct HodgeStar {};
struct InteriorProduct {};
struct Wedge {};

struct ExteriorDerivative {};
struct CodifferentialDerivative {};

template<size_type I> struct P_ExteriorDerivative {};
template<size_type I> struct P_CodifferentialDerivative {};

struct MapTo {};

struct Cross {};

struct Dot {};
}}//namespace calculus// // namespace tags


namespace ct=calculus::tags;

namespace traits
{
template<typename T> struct is_expression { static constexpr bool value = false; };

template<typename ...T, template<typename ...> class F>
struct is_expression<F<Expression<T...>>> { static constexpr bool value = true; };

template<typename T> struct is_primary_complex { static constexpr bool value = false; };
template<typename T> struct is_primary_complex<std::complex<T>>
{
    static constexpr bool value = std::is_arithmetic<T>::value;
};

template<typename T> struct is_primary_scalar
{
    static constexpr bool value = std::is_arithmetic<T>::value || is_primary_complex<T>::value;
};
template<typename T> using is_primary_scalar_t=  std::enable_if_t<is_primary_scalar<T>::value>;


template<typename T> struct is_primary
{
    static constexpr bool value = (is_primary_scalar<T>::value || is_ntuple<T>::value) && !(is_expression<T>::value);
};
template<typename T> using is_primary_t=  std::enable_if_t<is_primary<T>::value>;


//template<typename T> struct is_ntuple { static constexpr bool entity = false; };
//template<typename T, int ...N> struct is_ntuple<nTuple<T, N...>> { static constexpr bool entity = true; };
template<typename T> using is_primary_ntuple_t=std::enable_if_t<is_ntuple<T>::value && !(is_expression<T>::value)>;
template<typename T> using is_expression_ntuple_t=std::enable_if_t<is_ntuple<T>::value && (is_expression<T>::value)>;


template<typename> class dof;

template<typename TOP, typename T0, typename ... T>
struct dof<Expression<TOP, T0, T...> > : public dof<T0>::type {};


template<typename> class iform;

template<typename TOP, typename T0, typename ... T>
struct iform<Expression<TOP, T0, T...> > : public iform<T0>::type {};

template<typename T>
struct iform<Expression<ct::HodgeStar, T> > : public std::integral_constant<size_type, rank<T>::value - iform<T>::value>
{
};

template<typename T0, typename T1>
struct iform<Expression<ct::InteriorProduct, T0, T1> > : public std::integral_constant<size_type, iform<T1>::value - 1>
{
};

template<typename T>
struct iform<Expression<ct::ExteriorDerivative, T> > : public std::integral_constant<size_type, iform<T>::value + 1>
{
};

template<typename T>
struct iform<Expression<ct::CodifferentialDerivative, T> > : public std::integral_constant<size_type,
        iform<T>::value - 1>
{
};


template<typename T, size_type I>
struct iform<Expression<ct::P_ExteriorDerivative<I>, T> >
        : public std::integral_constant<size_type, iform<T>::value + 1>
{
};

template<typename T, size_type I>
struct iform<Expression<ct::P_CodifferentialDerivative<I>, T> >
        : public std::integral_constant<size_type, iform<T>::value - 1>
{
};

template<typename T0, typename T1>
struct iform<Expression<ct::Wedge, T0, T1> >
        : public std::integral_constant<size_type, iform<T0>::value + iform<T1>::value>
{
};
template<size_type I> struct iform<std::integral_constant<size_type, I> >
        : public std::integral_constant<size_type, I>
{
};

template<size_type I, typename T1>
struct iform<Expression<ct::MapTo, T1, std::integral_constant<size_type, I> > >
        : public std::integral_constant<size_type, I>
{
};


template<typename ...> struct value_type;

template<typename T>
struct value_type<Expression<ct::HodgeStar, T> > { typedef typename value_type<T>::type type; };

template<typename T>
struct value_type<Expression<ct::ExteriorDerivative, T> >
{
    typedef typename std::result_of<simpla::_impl::multiplies(
            typename geometry::traits::scalar_type<typename traits::mesh_type<T>::type>::type,
            typename value_type<T>::type)>::type type;
};

template<typename T>
struct value_type<Expression<ct::CodifferentialDerivative, T> >
{
    typedef typename std::result_of<
            simpla::_impl::multiplies(
                    typename geometry::traits::scalar_type<typename traits::mesh_type<T>::type>::type,
                    typename value_type<T>::type)>::type type;
};


template<typename T, size_type I>
struct value_type<Expression<ct::P_ExteriorDerivative<I>, T> >
{
    typedef typename std::result_of<simpla::_impl::multiplies(
            typename geometry::traits::scalar_type<typename traits::mesh_type<T>::type>::type,
            typename value_type<T>::type)>::type type;
};

template<typename T, size_type I>
struct value_type<Expression<ct::P_CodifferentialDerivative<I>, T> >
{
    typedef typename std::result_of<
            simpla::_impl::multiplies(typename geometry::traits::scalar_type<typename traits::mesh_type<T>::type>::type,
                                      typename value_type<T>::type)>::type type;
};


template<typename T0, typename T1>
struct value_type<Expression<ct::Wedge, T0, T1> >
{

    typedef typename std::result_of<simpla::_impl::multiplies(typename value_type<T0>::type,
                                                              typename value_type<T1>::type)>::type type;
};

template<typename T0, typename T1>
struct value_type<Expression<ct::InteriorProduct, T0, T1> >
{

    typedef typename std::result_of<simpla::_impl::multiplies(typename value_type<T0>::type,
                                                              typename value_type<T1>::type)>::type type;
};


template<typename T0, typename T1>
struct value_type<Expression<ct::Cross, T0, T1> >
{
    typedef typename value_type<T0>::type type;
};

template<typename T0, typename T1>
struct value_type<Expression<ct::Dot, T0, T1> > { typedef typename value_type<typename value_type<T0>::type>::type type; };


template<typename T, typename ...Others>
struct value_type<Expression<ct::MapTo, T, Others...> > { typedef typename value_type<T>::type type; };


} //namespace traits


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
template<typename T> inline Expression<ct::HodgeStar, T>
hodge_star(T const &f) { return Expression<ct::HodgeStar, T>(f); }

template<typename TL, typename TR> inline Expression<ct::Wedge, TL, TR>
wedge(TL const &l, TR const &r) { return Expression<ct::Wedge, TL, TR> > (l, r); };


template<typename TL, typename TR>
inline Expression<ct::InteriorProduct, TL, TR>
interior_product(TL const &l, TR const &r) { return Expression<ct::InteriorProduct, TL, TR>(l, r); };


//template<typename  T>
//inline auto operator*(T const &f) DECL_RET_TYPE((hodge_star(f)))
//
//template<size_type ndims, typename TL, typename ...T>
//inline auto iv(nTuple<TL, ndims> const &v, Field<T...> const &f) DECL_RET_TYPE((interior_product(v, f)))
//
//template<typename ...T1, typename ... T2>
//inline auto operator^(Field<T1...> const &lhs, Field<T2...> const &rhs) DECL_RET_TYPE((wedge(lhs, rhs)))

/** @} */

/**
 * @defgroup  vector_algebra   Linear algebra on vector fields
 * @{
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$  	            | negate operation
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$  	            | positive operation
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$ +\f$\Omega^n\f$ 	| add
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$ -\f$\Omega^n\f$ 	| subtract
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$ *Scalar  	    | multiply
 *  \f$\Omega^n\f$ = Scalar * \f$\Omega^n\f$  	    | multiply
 *  \f$\Omega^n\f$ = \f$\Omega^n\f$ / Scalar  	    | divide
 *
 */
template<typename TL, typename TR>
inline auto inner_product(TL const &lhs, TR const &rhs) DECL_RET_TYPE(wedge(lhs, hodge_star(rhs)));

template<typename ...TL, typename TR>
inline auto dot(TL const &lhs, TR const &rhs) DECL_RET_TYPE(wedge(lhs, hodge_star(rhs)));


template<typename TL, typename TR> inline auto
cross(TL const &lhs, TR const &rhs, ENABLE_IF((traits::iform<TL>::value == mesh::EDGE)))
DECL_RET_TYPE((wedge(lhs, rhs)))


template<typename TL, typename TR> inline auto
cross(TL const &lhs, TR const &rhs, ENABLE_IF((traits::iform<TL>::value == mesh::FACE)))
DECL_RET_TYPE(hodge_star(wedge(hodge_star(lhs), hodge_star(rhs))));


template<typename TL, typename TR> inline Expression<ct::Cross, TL, TR>
cross(TL const &l, TR const &r, ENABLE_IF((traits::iform<TL>::value == mesh::VERTEX)))
{
    return Expression<ct::Cross, TL, TR>(l, r);
};


template<typename TL, typename TR> inline Expression<ct::Dot, TL, TR>
dot(TL const &lhs, TR const &rhs, ENABLE_IF((traits::iform<TL>::value == mesh::VERTEX)))
{
    return Expression<ct::Dot, TL, TR>(lhs, rhs);
};


//template<typename TL, typename ... TR> inline auto
//dot(nTuple<TL, 3> const &v, Field<TR...> const &f) DECL_RET_TYPE((interior_product(v, f)))
//
//template<typename ...TL, typename TR> inline auto
//dot(Field<TL...> const &f, nTuple<TR, 3> const &v) DECL_RET_TYPE((interior_product(v, f)));
//
//template<typename ... TL, typename TR> inline auto
//cross(nTuple<TR, 3> const &v, Field<TL...> const &f) DECL_RET_TYPE((interior_product(v, hodge_star(f))));
//
//template<typename ... T, typename TL> inline auto
//cross(Field<T...> const &f, nTuple<TL, 3> const &v) DECL_RET_TYPE((interior_product(v, f)));

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

template<size_type I, typename U>
inline Expression<ct::MapTo, U, std::integral_constant<size_type, I> >
map_to(U const &f)
{
    return Expression<ct::MapTo, U, std::integral_constant<size_type, I >>(f, std::integral_constant<size_type, I>());
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

template<typename U>
inline Expression<ct::ExteriorDerivative, U>
exterior_derivative(U const &f) { return Expression<ct::ExteriorDerivative, U>(f); }

template<typename U>
inline Expression<ct::CodifferentialDerivative, U>
codifferential_derivative(U const &f) { return Expression<ct::CodifferentialDerivative, U>(f)); };
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
grad(T const &f, index_const <mesh::VERTEX>) DECL_RET_TYPE((exterior_derivative(f)))

template<typename T> inline auto
grad(T const &f, index_const <mesh::VOLUME>) DECL_RET_TYPE(((codifferential_derivative(-f))))

template<typename T> inline auto
grad(T const &f) DECL_RET_TYPE((grad(f, traits::iform<T>())))

template<typename T> inline auto
diverge(T const &f, index_const <mesh::FACE>) DECL_RET_TYPE((exterior_derivative(f)))

template<typename T> inline auto
diverge(T const &f, index_const <mesh::EDGE>) DECL_RET_TYPE((codifferential_derivative(-f)))

template<typename T> inline auto
diverge(T const &f) DECL_RET_TYPE((diverge(f, traits::iform<T>())))


template<typename T> inline auto
curl(T const &f, index_const <mesh::EDGE>) DECL_RET_TYPE((exterior_derivative(f)))

template<typename T> inline auto
curl(T const &f, index_const <mesh::FACE>) DECL_RET_TYPE((codifferential_derivative(-f)))

template<typename T> inline auto
curl(T const &f) DECL_RET_TYPE((curl(f, traits::iform<T>())))

template<size_type I, typename U>
inline Expression<ct::P_ExteriorDerivative<I>, U>
p_exterior_derivative(U const &f) { return Expression<ct::P_ExteriorDerivative < I>, U > (f); }


template<size_type I, typename U>
Expression<ct::P_CodifferentialDerivative<I>, U>
p_codifferential_derivative(U const &f) { return Expression<ct::P_CodifferentialDerivative < I>, U > (f); };


template<typename T> inline auto
curl_pdx(T const &f, index_const <mesh::EDGE>) DECL_RET_TYPE((p_exterior_derivative<0>(f)))

template<typename T> inline auto
curl_pdx(T const &f, index_const <mesh::FACE>) DECL_RET_TYPE((p_codifferential_derivative<0>(f)))

template<typename T> inline auto
curl_pdx(T const &f) DECL_RET_TYPE((curl_pdx(f, traits::iform<T>())))

template<typename T> inline auto
curl_pdy(T const &f, index_const <mesh::EDGE>) DECL_RET_TYPE((p_exterior_derivative<1>(f)))

template<typename T> inline auto
curl_pdy(T const &f, index_const <mesh::FACE>) DECL_RET_TYPE((p_codifferential_derivative<1>(f)))

template<typename T> inline auto
curl_pdy(T const &f) DECL_RET_TYPE((curl_pdy(f, traits::iform<T>())))

template<typename T> inline auto
curl_pdz(T const &f, index_const <mesh::EDGE>) DECL_RET_TYPE((p_exterior_derivative<2>(f)))

template<typename T> inline auto
curl_pdz(T const &f, index_const <mesh::FACE>) DECL_RET_TYPE((p_codifferential_derivative<2>(f)))

template<typename T> inline auto
curl_pdz(T const &f) DECL_RET_TYPE((curl_pdz(f, traits::iform<T>())))
/** @} */

/** @} */

}// namespace simpla

#endif /* CALCULUS_H_ */
