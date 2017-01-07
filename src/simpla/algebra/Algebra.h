//
// Created by salmon on 16-12-22.
//

#ifndef SIMPLA_ALGEBRACOMMON_H
#define SIMPLA_ALGEBRACOMMON_H

#include <simpla/SIMPLA_config.h>
#include <simpla/mpl/integer_sequence.h>
#include <simpla/mpl/type_traits.h>
#include <type_traits>
#include <utility>

namespace simpla {
enum { VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3, FIBER = 6 };

namespace algebra {

namespace declare {
template <typename...>
struct Expression;
template <typename, size_type...>
struct nTuple_;
template <typename, size_type NDIMS, bool SLOW_FIRST = true>
struct Array_;
template <typename, typename, size_type...>
struct Field_;
}
namespace calculus {
template <typename...>
struct calculator;
template <typename...>
struct expr_parser;
}

namespace traits {

template <typename T>
struct value_type {
    typedef T type;
};

template <typename T>
using value_type_t = typename value_type<T>::type;

template <typename T>
struct value_type<T&> {
    typedef T& type;
};
template <typename T>
struct value_type<T const&> {
    typedef T const& type;
};

template <typename T>
struct value_type<T*> {
    typedef T type;
};
template <typename T, size_type N>
struct value_type<T[N]> {
    typedef T type;
};
template <typename T>
struct value_type<T const*> {
    typedef T type;
};
template <typename T>
struct value_type<T const[]> {
    typedef T type;
};

template <typename T>
struct field_value_type {
    typedef T type;
};
template <typename T>
using field_value_t = typename field_value_type<T>::type;

template <typename T>
struct sub_type {
    typedef T type;
};
template <typename T>
using sub_type_t = typename sub_type<T>::type;

template <typename...>
struct pod_type;
template <typename... T>
using pod_type_t = typename pod_type<T...>::type;
template <typename T>
struct pod_type<T> {
    typedef T type;
};

template <typename T>
struct scalar_type {
    typedef Real type;
};

template <typename T>
using scalar_type_t = typename scalar_type<T>::type;

template <typename...>
struct is_complex : public std::integral_constant<bool, false> {};

template <typename T>
struct is_complex<std::complex<T>> : public std::integral_constant<bool, true> {};

template <typename...>
struct is_scalar : public std::integral_constant<bool, false> {};

template <typename T>
struct is_scalar<T>
    : public std::integral_constant<bool, std::is_arithmetic<std::decay_t<T>>::value ||
                                              is_complex<std::decay_t<T>>::value> {};
template <typename First, typename... Others>
struct is_scalar<First, Others...>
    : public std::integral_constant<bool, is_scalar<First>::value && is_scalar<Others...>::value> {
};

template <typename...>
struct is_field;

template <typename T>
struct is_field<T> : public std::integral_constant<bool, false> {};
template <typename First, typename... Others>
struct is_field<First, Others...>
    : public std::integral_constant<bool, is_field<First>::value || is_field<Others...>::value> {};
template <typename...>
struct is_array;

template <typename T>
struct is_array<T> : public std::integral_constant<bool, false> {};

template <typename First, typename... Others>
struct is_array<First, Others...>
    : public std::integral_constant<bool, (is_array<First>::value && !is_field<First>::value) ||
                                              is_array<Others...>::value> {};

template <typename...>
struct is_nTuple;

template <typename T>
struct is_nTuple<T> : public std::integral_constant<bool, false> {};

template <typename First, typename... Others>
struct is_nTuple<First, Others...>
    : public std::integral_constant<bool, (is_nTuple<First>::value &&
                                           !(is_field<Others...>::value || is_array<Others...>::value)) ||
                                              is_nTuple<Others...>::value> {};
template <typename...>
struct is_expression;

template <typename T>
struct is_expression<T> : public std::integral_constant<bool, false> {};

template <typename... T>
struct is_expression<declare::Expression<T...>> : public std::integral_constant<bool, true> {};

template <typename T>
struct reference {
    typedef T type;
};
template <typename T>
using reference_t = typename reference<T>::type;
template <typename T, int N>
struct reference<T[N]> {
    typedef T* type;
};
template <typename T, int N>
struct reference<const T[N]> {
    typedef T const* type;
};

//***********************************************************************************************************************

template <typename>
struct iform : public index_const<VERTEX> {};
template <typename T>
struct iform<const T> : public iform<T> {};

template <typename>
struct dof : public index_const<1> {};
template <typename T>
struct dof<const T> : public dof<T> {};

template <typename>
struct rank : public index_const<3> {};
template <typename T>
struct rank<const T> : public rank<T> {};

template <typename>
struct extent : public index_const<0> {};

template <typename T>
struct extents : public index_sequence<> {};

//**************************************************************************
// From nTuple.h

template <typename T, size_type... I0>
struct reference<declare::nTuple_<T, I0...>> {
    typedef declare::nTuple_<T, I0...>& type;
};

template <typename T, size_type... I0>
struct reference<const declare::nTuple_<T, I0...>> {
    typedef declare::nTuple_<T, I0...> const& type;
};

template <typename T, size_type... I>
struct rank<declare::nTuple_<T, I...>> : public index_const<sizeof...(I)> {};

template <typename V, size_type... I>
struct extents<declare::nTuple_<V, I...>> : public index_sequence<I...> {};

template <typename V, size_type I0, size_type... I>
struct extent<declare::nTuple_<V, I0, I...>> : public index_const<I0> {};

template <typename T, size_type I0>
struct value_type<declare::nTuple_<T, I0>> {
    typedef T type;
};

template <typename T, size_type... I>
struct value_type<declare::nTuple_<T, I...>> {
    typedef T type;
};

template <typename T, size_type I0, size_type... I>
struct sub_type<declare::nTuple_<T, I0, I...>> {
    typedef std::conditional_t<sizeof...(I) == 0, T, declare::nTuple_<T, I...>> type;
};

template <typename T, size_type I0>
struct pod_type<declare::nTuple_<T, I0>> {
    typedef T type[I0];
};

template <typename T, size_type I0, size_type... I>
struct pod_type<declare::nTuple_<T, I0, I...>> {
    typedef typename pod_type<declare::nTuple_<T, I...>>::type type[I0];
};

template <typename, typename>
struct make_nTuple;

template <typename TV, size_type... I>
struct make_nTuple<TV, simpla::index_sequence<I...>> {
    typedef declare::nTuple_<TV, I...> type;
};

template <typename TV>
struct make_nTuple<TV, simpla::index_sequence<>> {
    typedef TV type;
};

template <typename...>
struct nTuple_traits;
template <typename T>
struct nTuple_traits<T> {
    typedef typename make_nTuple<value_type_t<T>, extents<T>>::type type;
    typedef calculus::calculator<type> calculator;
};

//**************************************************************************
// From Field.h

template <typename>
struct mesh_type {
    typedef void type;
};

template <typename TV, typename TM, size_type... I>
struct mesh_type<declare::Field_<TV, TM, I...>> {
    typedef TM type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct is_field<declare::Field_<TV, TM, IFORM, DOF>> : public std::integral_constant<bool, true> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<declare::Field_<TV, TM, IFORM, DOF>> {
    typedef declare::Field_<TV, TM, IFORM, DOF> const& type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<const declare::Field_<TV, TM, IFORM, DOF>> {
    typedef declare::Field_<TV, TM, IFORM, DOF> const& type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct value_type<declare::Field_<TV, TM, IFORM, DOF>> {
    typedef TV type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct rank<declare::Field_<TV, TM, IFORM, DOF>> : public index_const<3> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct iform<declare::Field_<TV, TM, IFORM, DOF>> : public index_const<IFORM> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct dof<declare::Field_<TV, TM, IFORM, DOF>> : public index_const<DOF> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct field_value_type<declare::Field_<TV, TM, IFORM, DOF>> {
    typedef std::conditional_t<DOF == 1, TV, declare::nTuple_<TV, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple,
                               declare::nTuple_<cell_tuple, 3>>
        type;
};

template <typename V>
struct field_traits {
    typedef declare::Field_<V, typename mesh_type<V>::type, iform<V>::value, dof<V>::value> type;
    typedef calculus::calculator<type> calculator;
};
//
// template <typename T, typename Enable = void>
// struct calculator_selector {};
//
// template <typename V>
// struct calculator_selector<V, std::enable_if_t<is_field<V>::value>> {
//    typedef declare::Field_<value_type_t<V>, typename mesh_type<V>::type, iform<V>::value,
//                            dof<V>::value>
//        field_type;
//    typedef calculus::calculator<field_type> type;
//};
// template <typename T>
// struct calculator_selector<T, std::enable_if_t<is_nTuple<T>::value>> {
//    typedef typename make_nTuple<value_type_t<T>, extents<T>>::type nTuple_type;
//    typedef calculus::calculator<nTuple_type> type;
//};

//**************************************************************************

}  // namespace traits

}  // namespace algebra

template <typename T, size_type... N>
using nTuple = algebra::declare::nTuple_<T, N...>;

template <typename T, size_type N>
using Vector = algebra::declare::nTuple_<T, N>;

template <typename T, size_type M, size_type N>
using Matrix = algebra::declare::nTuple_<T, M, N>;

template <typename T, size_type... N>
using Tensor = algebra::declare::nTuple_<T, N...>;

template <typename T, size_type N, bool is_slow_first = true>
using Array = algebra::declare::Array_<T, N, is_slow_first>;

}  // namespace simpla
#endif  // SIMPLA_ALGEBRACOMMON_H
