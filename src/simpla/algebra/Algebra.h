//
// Created by salmon on 16-12-22.
//

#ifndef SIMPLA_ALGEBRACOMMON_H
#define SIMPLA_ALGEBRACOMMON_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/CheckConcept.h>
#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/integer_sequence.h>
#include <simpla/utilities/type_traits.h>
#include <utility>
namespace simpla {

namespace traits {
template <typename T>
struct is_primary_field : public std::integral_constant<bool, false> {};
template <typename TM, typename TV, int IFORM, int DOF>
class Field;
template <typename TM, typename TV, int IFORM, int DOF>
struct is_primary_field<Field<TM, TV, IFORM, DOF>> : public std::integral_constant<bool, true> {};

template <typename>
struct num_of_dimension : public int_const<3> {};

// CHECK_MEMBER_TYPE(value_type, value_type)
//
// template <typename T>
// using value_type_t = typename value_type<T>::type;
//
// template <typename T>
// struct value_type<T&> {
//    typedef T& type;
//};
// template <typename T>
// struct value_type<T const&> {
//    typedef T const& type;
//};
//
// template <typename T>
// struct value_type<T*> {
//    typedef T type;
//};
// template <typename T, int N>
// struct value_type<T[N]> {
//    typedef T type;
//};
// template <typename T>
// struct value_type<T const*> {
//    typedef T type;
//};
// template <typename T>
// struct value_type<T const[]> {
//    typedef T type;
//};
//
// template <typename T>
// struct reference {
//   private:
//    typedef std::true_type yes;
//    typedef std::false_type no;
//
//    template <typename U>
//    static auto test(int) -> typename U::prefer_pass_by_reference;
//    template <typename>
//    static no test(...);
//
//   public:
//    typedef std::conditional_t<std::is_same<decltype(test<T>(0)), no>::value, T, T&> type;
//};
// template <typename T>
// using reference_t = typename reference<T>::type;
// template <typename T, int N>
// struct reference<T[N]> {
//    typedef T* type;
//};
// template <typename T, int N>
// struct reference<const T[N]> {
//    typedef T const* type;
//};

//**************************************************************************************************

template <typename T>
struct field_value_type {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename U>
    static auto test(int) -> typename U::field_value_type;
    template <typename>
    static no test(...);
    typedef decltype(test<T>(0)) check_result;

   public:
    typedef std::conditional_t<std::is_same<check_result, no>::value, value_type_t<T>, check_result> type;
};

template <typename T>
using field_value_t = typename field_value_type<T>::type;

//CHECK_BOOLEAN_TYPE_MEMBER(is_array, is_array)
//CHECK_BOOLEAN_TYPE_MEMBER(is_field, is_field)
//// CHECK_BOOLEAN_TYPE_MEMBER(is_nTuple, is_nTuple)
////CHECK_BOOLEAN_TYPE_MEMBER(is_expression, is_expression)
//
//template <typename First, typename... Others>
//struct is_field<First, Others...>
//    : public std::integral_constant<bool, is_field<First>::value || is_field<Others...>::value> {};
//
//template <typename First, typename... Others>
//struct is_array<First, Others...>
//    : public std::integral_constant<bool,
//                                    (is_array<First>::value && !is_field<First>::value) || is_array<Others...>::value> {
//};

// template <typename First, typename... Others>
// struct is_nTuple<First, Others...>
//    : public std::integral_constant<bool,
//                                    (is_nTuple<First>::value &&
//                                     !(is_field<Others...>::value || is_array<Others...>::value)) ||
//                                        is_nTuple<Others...>::value> {};

CHECK_STATIC_INTEGRAL_CONSTEXPR_DATA_MEMBER(ndims, NDIMS, 1)

CHECK_STATIC_INTEGRAL_CONSTEXPR_DATA_MEMBER(dof, dof, 1)

CHECK_STATIC_INTEGRAL_CONSTEXPR_DATA_MEMBER(iform, iform, VERTEX)

CHECK_STATIC_INTEGRAL_CONSTEXPR_DATA_MEMBER(num_of_sub, num_of_sub, 1)

template <typename T>
struct iform<const T> : public int_const<iform<T>::value> {};

// template <typename _T>
// struct iform_ {
//   private:
//    template <typename U>
//    static auto test(int) -> std::integral_constant<int, U::GetIFORM>;
//    template <typename>
//    static std::integral_constant<int, 0> test(...);
//
//   public:
//    static constexpr int value = decltype(test<_T>(0))::value;
//};
// template <typename T>
// struct GetIFORM : public int_const<iform_<T>::value> {};
//
// template <typename _T>
// struct dof_ {
//   private:
//    template <typename U>
//    static auto test(int) -> std::integral_constant<int, U::GetDOF>;
//    template <typename>
//    static std::integral_constant<int, 1> test(...);
//
//   public:
//    static constexpr int value = decltype(test<_T>(0))::value;
//};
// template <typename T>
// struct GetDOF : public int_const<dof_<T>::value> {};
// template <typename T>
// struct GetDOF<const T> : public GetDOF<T> {};
//
// template <typename>
// struct rank : public int_const<3> {};
// template <typename T>
// struct rank<const T> : public rank<T> {};
//
// template <typename>
// struct extent : public int_const<0> {};
// template <typename T>
// struct extent<const T> : public int_const<extent<T>::value> {};
// template <typename T>
// struct extents : public int_sequence<> {};
//
// template <typename T>
// struct scalar_type {
//    typedef Real type;
//};
//
// template <typename T>
// using scalar_type_t = typename scalar_type<T>::type;
//
// template <typename...>
// struct is_complex : public std::integral_constant<bool, false> {};
//
// template <typename T>
// struct is_complex<std::complex<T>> : public std::integral_constant<bool, true> {};
//
// template <typename...>
// struct is_scalar : public std::integral_constant<bool, false> {};
//
// template <typename T>
// struct is_scalar<T>
//    : public std::integral_constant<bool,
//                                    std::is_arithmetic<std::decay_t<T>>::value || is_complex<std::decay_t<T>>::value>
//                                    {
//};
// template <typename First, typename... Others>
// struct is_scalar<First, Others...>
//    : public std::integral_constant<bool, is_scalar<First>::value && is_scalar<Others...>::value> {};

}  // namespace traits

// template <typename T, int... N>
// using nTuple = algebra::declare::nTuple_<T, N...>;
//
// template <typename T, int N>
// using Vector = algebra::declare::nTuple_<T, N>;
//
// template <typename T, int M, int N>
// using Matrix = algebra::declare::nTuple_<T, M, N>;
//
// template <typename T, int... N>
// using Tensor = algebra::declare::nTuple_<T, N...>;
//
// template <typename T, int N, bool is_slow_first = true>
// using Array = algebra::declare::Array_<T, N, is_slow_first>;

}  // namespace simpla
#endif  // SIMPLA_ALGEBRACOMMON_H
