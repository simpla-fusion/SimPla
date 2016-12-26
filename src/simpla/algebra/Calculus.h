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

#include <simpla/mpl/port_cxx14.h>
#include <simpla/mpl/macro.h>
#include <simpla/mpl/type_traits.h>
#include <simpla/mpl/integer_sequence.h>

#include "Algebra.h"
#include "Arithmetic.h"
//#include "Expression.h"

namespace simpla { namespace algebra
{
template<size_type I> using index_const=std::integral_constant<size_type, I>;

namespace declare { template<typename ...> class Expression; }
/**
 * @defgroup algebra Algebra
 * @ingroup algebra
 * @{
 **/

namespace tags
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
}  // namespace tags
namespace traits
{
template<typename T>
struct iform<declare::Expression<tags::HodgeStar, T> > : public index_const<rank<T>::value - iform<T>::value> {};

template<typename T0, typename T1>
struct iform<declare::Expression<tags::InteriorProduct, T0, T1> > : public index_const<traits::iform<T1>::value - 1> {};

template<typename T>
struct iform<declare::Expression<tags::ExteriorDerivative, T> > : public index_const<iform<T>::value + 1> {};

template<typename T>
struct iform<declare::Expression<tags::CodifferentialDerivative, T> > : public index_const<iform<T>::value - 1> {};


template<typename T, size_type I>
struct iform<declare::Expression<tags::P_ExteriorDerivative<I>, T> > : public index_const<iform<T>::value + 1> {};

template<typename T, size_type I>
struct iform<declare::Expression<tags::P_CodifferentialDerivative<I>, T> > : public index_const<iform<T>::value - 1> {};

template<typename T0, typename T1>
struct iform<declare::Expression<tags::Wedge, T0, T1> > : public index_const<iform<T0>::value + iform<T1>::value> {};

//template<size_type I>
//struct iform<index_const < I> > : public index_const <I>
//{
//};

//template<typename T1, size_type I>
//struct iform<declare::Expression<tags::MapTo, T1, index_const < I> > > : public index_const <I>
//{
//};


template<typename T, typename ...Others>
struct value_type<declare::Expression<tags::MapTo, T, Others...> > { typedef value_type_t <T> type; };


template<typename T>
struct value_type<declare::Expression<tags::HodgeStar, T> > { typedef value_type_t <T> type; };

template<typename T>
struct value_type<declare::Expression<tags::ExteriorDerivative, T> >
{
    typedef std::result_of_t<tags::multiplies(typename scalar_type<T>::type, value_type_t <T>)> type;
};

template<typename T>
struct value_type<declare::Expression<tags::CodifferentialDerivative, T> >
{
    typedef std::result_of_t<tags::multiplies(typename scalar_type<T>::type, value_type_t <T>)> type;
};


template<typename T, size_type I>
struct value_type<declare::Expression<tags::P_ExteriorDerivative<I>, T> >
{
    typedef std::result_of_t<tags::multiplies(typename scalar_type<T>::type, value_type_t <T>)> type;

};

template<typename T, size_type I>
struct value_type<declare::Expression<tags::P_CodifferentialDerivative<I>, T> >
{
    typedef std::result_of_t<tags::multiplies(typename scalar_type<T>::type, typename scalar_type<T>::type)> type;
};


template<typename T0, typename T1>
struct value_type<declare::Expression<tags::Wedge, T0, T1> >
{

    typedef std::result_of_t<tags::multiplies(scalar_type_t < T0 > , value_type_t < T1 > )> type;

};

template<typename T0, typename T1>
struct value_type<declare::Expression<tags::InteriorProduct, T0, T1> >
{
    typedef std::result_of_t<tags::multiplies(scalar_type_t < T0 > , value_type_t < T1 > )> type;
};


template<typename T0, typename T1>
struct value_type<declare::Expression<tags::Cross, T0, T1> >
{
    typedef std::result_of_t<tags::multiplies(scalar_type_t < T0 > , value_type_t < T1 > )> type;
};

template<typename T0, typename T1>
struct value_type<declare::Expression<tags::Dot, T0, T1> >
{
    typedef std::result_of_t<tags::multiplies(scalar_type_t < T0 > , value_type_t < T1 > )> type;
};


} //namespace traits

template<size_type I, typename U>
inline declare::Expression<tags::MapTo, U, index_const<I>>
map_to(U const &f)
{
    return declare::Expression<tags::MapTo, U, index_const<I> >(f, index_const<I>());
}

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
template<typename T> inline declare::Expression<tags::HodgeStar, T>
hodge_star(T const &f) { return declare::Expression<tags::HodgeStar, T>(f); }

template<typename TL, typename TR> inline declare::Expression<tags::Wedge, TL, TR>
wedge(TL const &l, TR const &r) { return declare::Expression<tags::Wedge, TL, TR>(l, r); };


template<typename TL, typename TR>
inline declare::Expression<tags::InteriorProduct, TL, TR>
interior_product(TL const &l, TR const &r) { return declare::Expression<tags::InteriorProduct, TL, TR>(l, r); };


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

template<typename TL, typename TR>
inline auto dot(TL const &lhs, TR const &rhs) DECL_RET_TYPE(wedge(lhs, hodge_star(rhs)));


template<typename TL, typename TR> inline auto
cross(TL const &lhs, TR const &rhs,
      ENABLE_IF((traits::iform<TL>::value == EDGE))) DECL_RET_TYPE((wedge(lhs, rhs)))


template<typename TL, typename TR> inline auto
cross(TL const &lhs, TR const &rhs, ENABLE_IF((traits::iform<TL>::value == FACE))) DECL_RET_TYPE(
        hodge_star(wedge(hodge_star(lhs), hodge_star(rhs))));


template<typename TL, typename TR> inline declare::Expression<tags::Cross, TL, TR>
cross(TL const &l, TR const &r, ENABLE_IF((traits::iform<TL>::value == VERTEX)))
{
    return declare::Expression<tags::Cross, TL, TR>(l, r);
};


template<typename TL, typename TR> inline declare::Expression<tags::Dot, TL, TR>
dot(TL const &lhs, TR const &rhs, ENABLE_IF((traits::iform<TL>::value == VERTEX)))
{
    return declare::Expression<tags::Dot, TL, TR>(lhs, rhs);
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
inline declare::Expression<tags::ExteriorDerivative, U>
exterior_derivative(U const &f) { return declare::Expression<tags::ExteriorDerivative, U>(f); }

template<typename U>
inline declare::Expression<tags::CodifferentialDerivative, U>
codifferential_derivative(U const &f) { return declare::Expression<tags::CodifferentialDerivative, U>(f); };
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
grad(T const &f, index_const<VERTEX>) DECL_RET_TYPE((exterior_derivative(f)))

template<typename T> inline auto
grad(T const &f, index_const<VOLUME>) DECL_RET_TYPE(((codifferential_derivative(-f))))

template<typename T> inline auto
grad(T const &f) DECL_RET_TYPE((grad(f, traits::iform<T>())))

template<typename T> inline auto
diverge(T const &f, index_const<FACE>) DECL_RET_TYPE((exterior_derivative(f)))

template<typename T> inline auto
diverge(T const &f, index_const<EDGE>) DECL_RET_TYPE((codifferential_derivative(-f)))

template<typename T> inline auto
diverge(T const &f) DECL_RET_TYPE((diverge(f, traits::iform<T>())))


template<typename T> inline auto
curl(T const &f, index_const<EDGE>) DECL_RET_TYPE((exterior_derivative(f)))

template<typename T> inline auto
curl(T const &f, index_const<FACE>) DECL_RET_TYPE((codifferential_derivative(-f)))

template<typename T> inline auto
curl(T const &f) DECL_RET_TYPE((curl(f, traits::iform<T>())))

template<size_type I, typename U>
inline declare::Expression<tags::P_ExteriorDerivative<I>, U>
p_exterior_derivative(U const &f) { return declare::Expression<tags::P_ExteriorDerivative<I>, U>(f); }


template<size_type I, typename U>
declare::Expression<tags::P_CodifferentialDerivative<I>, U>
p_codifferential_derivative(U const &f) { return declare::Expression<tags::P_CodifferentialDerivative<I>, U>(f); };


template<typename T> inline auto
curl_pdx(T const &f, index_const<EDGE>) DECL_RET_TYPE((p_exterior_derivative<0>(f)))

template<typename T> inline auto
curl_pdx(T const &f, index_const<FACE>) DECL_RET_TYPE((p_codifferential_derivative<0>(f)))

template<typename T> inline auto
curl_pdx(T const &f) DECL_RET_TYPE((curl_pdx(f, traits::iform<T>())))

template<typename T> inline auto
curl_pdy(T const &f, index_const<EDGE>) DECL_RET_TYPE((p_exterior_derivative<1>(f)))

template<typename T> inline auto
curl_pdy(T const &f, index_const<FACE>) DECL_RET_TYPE((p_codifferential_derivative<1>(f)))

template<typename T> inline auto
curl_pdy(T const &f) DECL_RET_TYPE((curl_pdy(f, traits::iform<T>())))

template<typename T> inline auto
curl_pdz(T const &f, index_const<EDGE>) DECL_RET_TYPE((p_exterior_derivative<2>(f)))

template<typename T> inline auto
curl_pdz(T const &f, index_const<FACE>) DECL_RET_TYPE((p_codifferential_derivative<2>(f)))

template<typename T> inline auto
curl_pdz(T const &f) DECL_RET_TYPE((curl_pdz(f, traits::iform<T>())))
/** @} */

/** @} */

}}// namespace simpla

#endif /* CALCULUS_H_ */
