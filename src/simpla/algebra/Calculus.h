/**
 * @file calculus.h
 *
 *  Created on: 2014-9-23
 *      Author: salmon
 */

#ifndef CALCULUS_H_
#define CALCULUS_H_

#include "simpla/algebra/ExpressionTemplate.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/integer_sequence.h"
#include "simpla/utilities/type_traits.h"

namespace simpla {
template <typename...>
class Expression;

namespace traits {

template <typename T>
struct iform : public std::integral_constant<int, NODE> {};

template <typename TOP, typename... Args>
struct iform<Expression<TOP, Args...>> : public std::integral_constant<int, max(iform<Args>::value...)> {};

template <typename TF>
struct dof : public std::integral_constant<int, 1> {};

template <typename>
struct value_type;

}  // namespace traits {

/**
 * @defgroup algebra Algebra
 * @ingroup algebra
 * @{
 **/

#define _SP_DEFINE_EXPR_BINARY_FUNCTION(_NAME_)                                            \
    namespace tags {                                                                       \
    struct _NAME_ {};                                                                      \
    }                                                                                      \
    template <typename T1, typename T2>                                                    \
    auto _NAME_(T1 const& l, T2 const& r) {                                                \
        return (Expression<tags::_NAME_, std::remove_reference_t<traits::reference_t<T1>>, \
                           std::remove_reference_t<traits::reference_t<T2>>>(l, r));       \
    }
#define DEF_BI_FUN(_NAME_)

#define _SP_DEFINE_EXPR_UNARY_FUNCTION(_NAME_)                                                \
    namespace tags {                                                                          \
    struct _NAME_ {};                                                                         \
    }                                                                                         \
    template <typename T1>                                                                    \
    auto _NAME_(T1 const& l) {                                                                \
        return Expression<tags::_NAME_, std::remove_reference_t<traits::reference_t<T1>>>(l); \
    }

/**
 * @defgroup exterior_algebra Exterior algebra on forms
 * @{
 *
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{N-n}\f$ =HodgeStar(\f$\Omega^n\f$ )	| hodge star, abbr. operator *
 *  \f$\Omega^{m+n}\f$ =wedge(\f$\Omega^m\f$ ,\f$\Omega^m\f$  )	| wedge product, abbr. operator^
 *  \f$\Omega^{n-1}\f$ =InteriorProduct(Vector field ,\f$\Omega^n\f$  )	|  interior product, abbr. iv
 *  \f$\Omega^{N}\f$ =InnerProduct(\f$\Omega^m\f$ ,\f$\Omega^m\f$ ) | inner product,
 **/

_SP_DEFINE_EXPR_UNARY_FUNCTION(hodge_star)
_SP_DEFINE_EXPR_BINARY_FUNCTION(interior_product)
_SP_DEFINE_EXPR_BINARY_FUNCTION(wedge)

//_SP_DEFINE_EXPR_BINARY_FUNCTION(cross)
//_SP_DEFINE_EXPR_BINARY_FUNCTION(dot)

namespace traits {

//******************************************************

template <typename T>
struct iform<Expression<tags::hodge_star, T>>
    : public std::integral_constant<int, std::rank<T>::value - iform<T>::value> {};

template <typename T>
struct value_type<Expression<tags::hodge_star, T>> {
    typedef value_type_t<T> type;
};
//******************************************************

template <typename T0, typename T1>
struct iform<Expression<tags::interior_product, T0, T1>> : public std::integral_constant<int, iform<T1>::value - 1> {};

template <typename T0, typename T1>
struct value_type<Expression<tags::interior_product, T0, T1>> {
    typedef std::result_of_t<tags::multiplication(value_type_t<T0>, value_type_t<T1>)> type;
};

//******************************************************

template <typename T0, typename T1>
struct iform<Expression<tags::wedge, T0, T1>>
    : public std::integral_constant<int, iform<T0>::value + iform<T1>::value> {};
template <typename T0, typename T1>
struct value_type<Expression<tags::wedge, T0, T1>> {
    typedef std::result_of_t<tags::multiplication(value_type_t<T0>, value_type_t<T1>)> type;
};
//******************************************************`

template <typename T0, typename T1>
struct iform<Expression<tags::cross, T0, T1>>
    : public std::integral_constant<int, (iform<T0>::value + iform<T1>::value) % 3> {};

template <typename T0, typename T1>
struct value_type<Expression<tags::cross, T0, T1>> {
    typedef std::result_of_t<tags::multiplication(value_type_t<T0>, value_type_t<T1>)> type;
};
//******************************************************

template <typename T0, typename T1>
struct iform<Expression<tags::dot, T0, T1>> : public std::integral_constant<int, NODE> {};

template <typename T0, typename T1>
struct value_type<Expression<tags::dot, T0, T1>> {
    typedef std::result_of_t<tags::multiplication(value_type_t<T0>, value_type_t<T1>)> type;
};

}  // namespace traits

template <typename TL, typename TR>
auto inner_product(TL const& lhs, TR const& rhs,
                   ENABLE_IF((traits::dimension<TL>::value + traits::dimension<TR>::value > 0))) {
    return (wedge(lhs, hodge_star(rhs)));
}

/**
 * @defgroup dif_calculus_form Differential calculus on forms
 * @{
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{n-1}\f$ =ExteriorDerivative(\f$\Omega^n\f$ )	| Exterior  Derivative, abbr. d
 *  \f$\Omega^{n+1}\f$ =Codifferential(\f$\Omega^n\f$ )	    | Codifferential Derivative, abbr. delta
 *
 */

_SP_DEFINE_EXPR_UNARY_FUNCTION(codifferential_derivative)

_SP_DEFINE_EXPR_UNARY_FUNCTION(exterior_derivative)

namespace tags {
template <int I>
struct p_exterior_derivative {};
template <int I>
struct p_codifferential_derivative {};

struct grad {};
struct curl {};
struct diverge {};
}  // namespace tags

namespace traits {

//******************************************************

template <typename T>
struct iform<Expression<tags::exterior_derivative, T>> : public std::integral_constant<int, iform<T>::value + 1> {};

template <typename T>
struct value_type<Expression<tags::exterior_derivative, T>> {
    typedef std::result_of_t<tags::multiplication(scalar_type_t<T>, value_type_t<T>)> type;
};

//******************************************************

template <typename T>
struct iform<Expression<tags::codifferential_derivative, T>> : public std::integral_constant<int, iform<T>::value - 1> {
};
template <typename T>
struct value_type<Expression<tags::codifferential_derivative, T>> {
    typedef std::result_of_t<tags::multiplication(typename scalar_type<T>::type, value_type_t<T>)> type;
};

//******************************************************
template <typename T, int I>
struct iform<Expression<tags::p_exterior_derivative<I>, T>> : public std::integral_constant<int, iform<T>::value + 1> {
};

template <typename T, int I>
struct value_type<Expression<tags::p_exterior_derivative<I>, T>> {
    typedef std::result_of_t<tags::multiplication(typename scalar_type<T>::type, value_type_t<T>)> type;
};
//******************************************************

template <typename T, int I>
struct iform<Expression<tags::p_codifferential_derivative<I>, T>>
    : public std::integral_constant<int, iform<T>::value - 1> {};
template <typename T, int I>
struct value_type<Expression<tags::p_codifferential_derivative<I>, T>> {
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
 *  R &=& 1-\sum_{s}\frac{\omega_{ps}^{2}}{\omega\left(\omega+\Omega_{s}\right)}\\
 *  L &=& 1+\sum_{s}\frac{\omega_{ps}^{2}}{\omega\left(\omega-\Omega_{s}\right)}\\
 *  P &=& 1-\sum_{s}\frac{\omega_{ps}^{2}}{\omega^{2}}
 *  \f}
 */
namespace tags {
template <int I>
struct map_to {};
};

namespace traits {

template <int I, typename T, typename... Others>
struct value_type<Expression<tags::map_to<I>, T, Others...>> {
    typedef value_type_t<T> type;
};
//******************************************************
template <int I, typename T0>
struct iform<Expression<tags::map_to<I>, T0>> : public std::integral_constant<int, I> {};
}  // namespace traits

template <int I, typename T1>
auto map_to(T1 const& l) {
    return (Expression<tags::map_to<I>, const T1>(l));
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
auto grad(T const& f, std::integral_constant<int, NODE>) {
    return ((exterior_derivative(f)));
}

template <typename T>
auto grad(T const& f, std::integral_constant<int, CELL>) {
    return (((codifferential_derivative(-f))));
}

template <typename T, int I>
auto grad(T const& f, std::integral_constant<int, I>) {
    return ((Expression<tags::grad, std::remove_reference_t<traits::reference_t<T>>>(f)));
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
    return ((Expression<tags::diverge, std::remove_reference_t<traits::reference_t<T>>>(f)));
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

// template <typename T>
// auto curl(T const& f, std::integral_constant<int, NODE> const&) {
//    return Expression<tags::curl, const std::remove_reference_t<traits::reference_t<T>>>(f);
//}
// template <typename T>
// auto curl(T const& f, std::integral_constant<int, CELL> const&) {
//    return Expression<tags::curl, const std::remove_reference_t<traits::reference_t<T>>>(f);
//}
template <typename T>
auto curl(T const& f) {
    return curl(f, traits::iform<std::remove_cv_t<T>>());
}

template <int I, typename U>
auto p_exterior_derivative(U const& f) {
    return Expression<tags::p_exterior_derivative<I>, std::remove_reference_t<traits::reference_t<U>>>(f);
}

template <int I, typename U>
auto p_codifferential_derivative(U const& f) {
    return ((Expression<tags::p_codifferential_derivative<I>, const std::remove_reference_t<traits::reference_t<U>>>(f)));
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
