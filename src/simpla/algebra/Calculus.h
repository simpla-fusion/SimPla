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

#include <simpla/utilities/ExpressionTemplate.h>
#include <simpla/utilities/Log.h>
#include <simpla/utilities/integer_sequence.h>
#include <simpla/utilities/macro.h>
#include <simpla/utilities/type_traits.h>

#include "Algebra.h"

namespace simpla {
template <typename...>
class Expression;

template <typename TM, typename TV, int ...>
class Field;
}

namespace std {
template <typename TM, typename TV, int IFORM, int DOF>
struct rank<simpla::Field<TM, TV, IFORM, DOF>> : public std::integral_constant<int, rank<TM>::value> {};

}  // namespace std{

namespace simpla {
namespace traits {

template <typename TM, typename TV, int IFORM, int DOF>
struct iform<Field<TM, TV, IFORM, DOF>> : public std::integral_constant<int, IFORM> {};

template <typename TM, typename TV, int IFORM, int DOF>
struct dof<Field<TM, TV, IFORM, DOF>> : public std::integral_constant<int, DOF> {};

template <typename TM, typename TV, int IFORM, int DOF>
struct value_type<Field<TM, TV, IFORM, DOF>> {
    typedef TV type;
};
}  // namespace traits {

/**
 * @defgroup algebra Algebra
 * @ingroup algebra
 * @{
 **/

#define _SP_DEFINE_EXPR_BINARY_FUNCTION(_NAME_)                                        \
    namespace tags {                                                                   \
    struct _##_NAME_ {};                                                               \
    }                                                                                  \
    template <typename T1, typename T2>                                                \
    Expression<tags::_##_NAME_, const T1, const T2> _NAME_(T1 const& l, T2 const& r) { \
        return (Expression<tags::_##_NAME_, const T1, const T2>(l, r));                \
    }
#define DEF_BI_FUN(_NAME_)

#define _SP_DEFINE_EXPR_UNARY_FUNCTION(_NAME_)                  \
    namespace tags {                                            \
    struct _##_NAME_ {};                                        \
    }                                                           \
    template <typename T1>                                      \
    Expression<tags::_##_NAME_, const T1> _NAME_(T1 const& l) { \
        return Expression<tags::_##_NAME_, const T1>(l);        \
    }

/**
 * @defgroup exterior_algebra Exterior algebra on forms
 * @{
 *
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{N-n}\f$ =HodgeStar(\f$\Omega^n\f$ )	| hodge star, abbr.
 operator *
 *  \f$\Omega^{m+n}\f$ =wedge(\f$\Omega^m\f$ ,\f$\Omega^m\f$  )	| wedge
 product, abbr. operator^
 *  \f$\Omega^{n-1}\f$ =InteriorProduct(Vector field ,\f$\Omega^n\f$  )	|
 interior product, abbr. iv
 *  \f$\Omega^{N}\f$ =InnerProduct(\f$\Omega^m\f$ ,\f$\Omega^m\f$ ) | inner
 product,

 **/

_SP_DEFINE_EXPR_UNARY_FUNCTION(hodge_star)

_SP_DEFINE_EXPR_BINARY_FUNCTION(interior_product)

_SP_DEFINE_EXPR_BINARY_FUNCTION(wedge)

//_SP_DEFINE_EXPR_BINARY_FUNCTION(cross)
//
//_SP_DEFINE_EXPR_BINARY_FUNCTION(dot)

namespace traits {

//******************************************************

template <typename T>
struct iform<Expression<tags::_hodge_star, T>>
    : public std::integral_constant<int, std::rank<T>::value - iform<T>::value> {};

template <typename T>
struct value_type<Expression<tags::_hodge_star, T>> {
    typedef value_type_t<T> type;
};
//******************************************************

template <typename T0, typename T1>
struct iform<Expression<tags::_interior_product, T0, T1>> : public std::integral_constant<int, iform<T1>::value - 1> {};

template <typename T0, typename T1>
struct value_type<Expression<tags::_interior_product, T0, T1>> {
    typedef std::result_of_t<tags::multiplication(value_type_t<T0>, value_type_t<T1>)> type;
};

//******************************************************

template <typename T0, typename T1>
struct iform<Expression<tags::_wedge, T0, T1>>
    : public std::integral_constant<int, iform<T0>::value + iform<T1>::value> {};
template <typename T0, typename T1>
struct value_type<Expression<tags::_wedge, T0, T1>> {
    typedef std::result_of_t<tags::multiplication(value_type_t<T0>, value_type_t<T1>)> type;
};
//******************************************************`

template <typename T0, typename T1>
struct iform<Expression<tags::_cross, T0, T1>>
    : public std::integral_constant<int, (iform<T0>::value + iform<T1>::value) % 3> {};

template <typename T0, typename T1>
struct value_type<Expression<tags::_cross, T0, T1>> {
    typedef std::result_of_t<tags::multiplication(value_type_t<T0>, value_type_t<T1>)> type;
};
//******************************************************

template <typename T0, typename T1>
struct iform<Expression<tags::_dot, T0, T1>> : public std::integral_constant<int, VERTEX> {};

template <typename T0, typename T1>
struct value_type<Expression<tags::_dot, T0, T1>> {
    typedef std::result_of_t<tags::multiplication(value_type_t<T0>, value_type_t<T1>)> type;
};
//******************************************************

}  // namespace traits

template <typename TL, typename TR>
auto inner_product(TL const& lhs, TR const& rhs,
                   ENABLE_IF((traits::dimension<TL>::value + traits::dimension<TR>::value > 0))) {
    return (wedge(lhs, hodge_star(rhs)));
}

// template <typename TL, typename TR>
// auto cross(TL const& lhs, TR const& rhs,
//           ENABLE_IF((traits::is_field<TL, TR>::value) &&
//                     (traits::iform<TL>::value == EDGE && traits::iform<TR>::value == EDGE))) {
//    return ((wedge(lhs, rhs)));
//}
//
// template <typename TL, typename TR>
// auto cross(TL const& lhs, TR const& rhs,
//           ENABLE_IF((traits::is_field<TL, TR>::value) &&
//                     (traits::iform<TL>::value == FACE && traits::iform<TR>::value == FACE))) {
//    return (hodge_star(wedge(hodge_star(lhs), hodge_star(rhs))));
//}

// template <typename TL, typename TR>
// auto dot(TL const& lhs, TR const& rhs,
//         ENABLE_IF((traits::is_field<TL, TR>::value) &&
//                   (traits::iform<TL>::value == VERTEX || traits::iform<TL>::value == VOLUME ||
//                    traits::iform<TR>::value == VERTEX || traits::iform<TR>::value == VOLUME))) {
//    return ((Expression<tags::_dot, const TL, const TR>(lhs, rhs)));
//};
//
// template <typename TL, typename TR>
// auto cross(TL const& l, TR const& r, ENABLE_IF(!(traits::is_field<TL, TR>::value))) {
//    return ((Expression<tags::_nTuple_cross, const TL, const TR>(l, r)));
//}
// template <typename TL, typename TR>
// auto dot(TL const& lhs, TR const& rhs, ENABLE_IF(!(traits::is_field<TL, TR>::value) &&
//                                                 !(std::is_arithmetic<TL>::value && std::is_arithmetic<TR>::value))) {
//    return Expression<tags::_nTuple_dot, const TL, const TR>(lhs, rhs);
//}
// template <typename TL, typename TR>
// auto dot(TL const& lhs, TR const& rhs, ENABLE_IF((std::is_arithmetic<TL>::value && std::is_arithmetic<TR>::value))) {
//    return lhs * rhs;
//}
// template<typename  T>
//  auto operator*(T const &f) AUTO_RETURN((hodge_star(f)))
//
// template<int ndims, typename TL, typename ...T>
//  auto iv(nTuple<TL, ndims> const &v, Field<T...> const &f)
// AUTO_RETURN((interior_product(v, f)))
//
// template<typename ...T1, typename ... T2>
//  auto operator^(Field<T1...> const &lhs, Field<T2...> const &rhs)
// AUTO_RETURN((wedge(lhs, rhs)))

// template<typename TL, typename ... TR>  auto
// dot(nTuple<TL, 3> const &v, Field<TR...> const &f)
// AUTO_RETURN((interior_product(v, f)))
//
// template<typename ...TL, typename TR>  auto
// dot(Field<TL...> const &f, nTuple<TR, 3> const &v)
// AUTO_RETURN((interior_product(v, f)));
//
// template<typename ... TL, typename TR>  auto
// cross(nTuple<TR, 3> const &v, Field<TL...> const &f)
// AUTO_RETURN((interior_product(v, hodge_star(f))));
//
// template<typename ... T, typename TL>  auto
// cross(Field<T...> const &f, nTuple<TL, 3> const &v)
// AUTO_RETURN((interior_product(v, f)));

/**
 * @defgroup dif_calculus_form Differential calculus on forms
 * @{
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{n-1}\f$ =ExteriorDerivative(\f$\Omega^n\f$ )	| Exterior
 * Derivative, abbr. d
 *  \f$\Omega^{n+1}\f$ =Codifferential(\f$\Omega^n\f$ )	| Codifferential
 * Derivative, abbr. delta
 *
 */

_SP_DEFINE_EXPR_UNARY_FUNCTION(codifferential_derivative)

_SP_DEFINE_EXPR_UNARY_FUNCTION(exterior_derivative)

namespace tags {
template <int I>
struct _p_exterior_derivative {};
template <int I>
struct _p_codifferential_derivative {};

struct _grad {};
struct _curl {};
struct _diverge {};
}  // namespace tags

// template<int I, typename T1> auto // Expression<tags::_p_exterior_derivative<I>,
// const T1>
// p_exterior_derivative(T1 const &l)
// AUTO_RETURN((Expression<tags::_p_exterior_derivative<I>, const T1>(l)))
//
//
// template<int I, typename T1> auto
// //Expression<tags::_p_codifferential_derivative<I>, const T1>
// p_codifferential_derivative(T1 const &l) AUTO_RETURN(
//        (Expression<tags::_p_codifferential_derivative<I>, const T1>(l)))

namespace traits {

//******************************************************

template <typename T>
struct iform<Expression<tags::_exterior_derivative, T>> : public std::integral_constant<int, iform<T>::value + 1> {};

template <typename T>
struct value_type<Expression<tags::_exterior_derivative, T>> {
    typedef std::result_of_t<tags::multiplication(scalar_type_t<T>, value_type_t<T>)> type;
};

//******************************************************

template <typename T>
struct iform<Expression<tags::_codifferential_derivative, T>>
    : public std::integral_constant<int, iform<T>::value - 1> {};
template <typename T>
struct value_type<Expression<tags::_codifferential_derivative, T>> {
    typedef std::result_of_t<tags::multiplication(typename scalar_type<T>::type, value_type_t<T>)> type;
};

//******************************************************
template <typename T, int I>
struct iform<Expression<tags::_p_exterior_derivative<I>, T>> : public std::integral_constant<int, iform<T>::value + 1> {
};

template <typename T, int I>
struct value_type<Expression<tags::_p_exterior_derivative<I>, T>> {
    typedef std::result_of_t<tags::multiplication(typename scalar_type<T>::type, value_type_t<T>)> type;
};
//******************************************************

template <typename T, int I>
struct iform<Expression<tags::_p_codifferential_derivative<I>, T>>
    : public std::integral_constant<int, iform<T>::value - 1> {};
template <typename T, int I>
struct value_type<Expression<tags::_p_codifferential_derivative<I>, T>> {
    typedef std::result_of_t<tags::multiplication(typename scalar_type<T>::type, typename scalar_type<T>::type)> type;
};

}  // namespace traits

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
 *  R &=&
 * 1-\sum_{s}\frac{\omega_{ps}^{2}}{\omega\left(\omega+\Omega_{s}\right)}\\
 *  L &=&
 * 1+\sum_{s}\frac{\omega_{ps}^{2}}{\omega\left(\omega-\Omega_{s}\right)}\\
 *  P &=& 1-\sum_{s}\frac{\omega_{ps}^{2}}{\omega^{2}}
 *  \f}
 */
namespace tags {
template <int I>
struct _map_to {};
};
namespace traits {

template <int I, typename T, typename... Others>
struct value_type<Expression<tags::_map_to<I>, T, Others...>> {
    typedef value_type_t<T> type;
};
//******************************************************
template <int I, typename T0>
struct iform<Expression<tags::_map_to<I>, T0>> : public std::integral_constant<int, I> {};
}  // namespace traits

template <int I, typename T1>
auto map_to(T1 const& l) {
    return (Expression<tags::_map_to<I>, const T1>(l));
}

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

/** @} */

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
template <typename T>
auto grad(T const& f, std::integral_constant<int, VERTEX>) {
    return ((exterior_derivative(f)));
}

template <typename T>
auto grad(T const& f, std::integral_constant<int, VOLUME>) {
    return (((codifferential_derivative(-f))));
}

template <typename T, int I>
auto grad(T const& f, std::integral_constant<int, I>) {
    return ((Expression<tags::_grad, const T>(f)));
}

template <typename T>
auto grad(T const& f) {
    return ((grad(f, traits::iform<T>())));
}

template <typename T>
auto diverge(T const& f, std::integral_constant<int, FACE> const&) {
    return ((exterior_derivative(f)));
}

template <typename T>
auto diverge(T const& f, std::integral_constant<int, EDGE> const&) {
    return ((-codifferential_derivative(f)));
}

template <typename T, int I>
auto diverge(T const& f, std::integral_constant<int, I> const&) {
    return ((Expression<tags::_diverge, const T>(f)));
}

template <typename T>
auto diverge(T const& f) {
    return ((diverge(f, traits::iform<T>())));
}

template <typename T>
auto curl(T const& f, std::integral_constant<int, EDGE> const&) {
    return ((exterior_derivative(f)));
}

template <typename T>
auto curl(T const& f, std::integral_constant<int, FACE> const&) {
    return ((-codifferential_derivative(f)));
}

template <typename T>
auto curl(T const& f, std::integral_constant<int, VERTEX> const&) {
    return Expression<tags::_curl, const T>(f);
}
template <typename T>
auto curl(T const& f, std::integral_constant<int, VOLUME> const&) {
    return Expression<tags::_curl, const T>(f);
}
template <typename T>
auto curl(T const& f) {
    return curl(f, traits::iform<T>());
}

template <int I, typename U>
auto p_exterior_derivative(U const& f) {
    return Expression<tags::_p_exterior_derivative<I>, U const>(f);
}

template <int I, typename U>
auto p_codifferential_derivative(U const& f) {
    return ((Expression<tags::_p_exterior_derivative<I>, U const>(f)));
}

template <typename T>
auto curl_pdx(T const& f, std::integral_constant<int, EDGE>) {
    return ((p_exterior_derivative<0>(f)));
}

template <typename T>
auto curl_pdx(T const& f, std::integral_constant<int, FACE>) {
    return ((p_codifferential_derivative<0>(f)));
}

template <typename T>
auto curl_pdx(T const& f) {
    return ((curl_pdx(f, traits::iform<T>())));
}

template <typename T>
auto curl_pdy(T const& f, std::integral_constant<int, EDGE>) {
    return ((p_exterior_derivative<1>(f)));
}

template <typename T>
auto curl_pdy(T const& f, std::integral_constant<int, FACE>) {
    return ((p_codifferential_derivative<1>(f)));
}

template <typename T>
auto curl_pdy(T const& f) {
    return ((curl_pdy(f, traits::iform<T>())));
}

template <typename T>
auto curl_pdz(T const& f, std::integral_constant<int, EDGE>) {
    return ((p_exterior_derivative<2>(f)));
}

template <typename T>
auto curl_pdz(T const& f, std::integral_constant<int, FACE>) {
    return ((p_codifferential_derivative<2>(f)));
}

template <typename T>
auto curl_pdz(T const& f) {
    return ((curl_pdz(f, traits::iform<T>())));
}
/** @} */

/** @} */
#undef _SP_DEFINE_EXPR_UNARY_FUNCTION
#undef _SP_DEFINE_EXPR_BINARY_FUNCTION

}  // namespace simpla//namespace algebra

#endif /* CALCULUS_H_ */
