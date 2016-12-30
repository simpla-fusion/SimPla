//
// Created by salmon on 16-12-22.
//

#ifndef SIMPLA_ALGEBRACOMMON_H
#define SIMPLA_ALGEBRACOMMON_H

#include <simpla/SIMPLA_config.h>
#include <type_traits>
#include <utility>
#include <simpla/mpl/type_traits.h>
#include <simpla/mpl/integer_sequence.h>

namespace simpla
{
enum { VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3, FIBER = 6 };

namespace algebra
{

namespace declare
{
template<typename ...> struct Expression;
template<typename, size_type ...> struct nTuple_;
template<typename, size_type NDIMS, bool SLOW_FIRST = true> struct Array_;
template<typename, typename, size_type ...> struct Field_;

}
namespace calculus { template<typename ...> struct calculator; }

namespace traits
{

template<typename> struct iform : public index_const<0> {};

template<typename> struct dof : public index_const<1> {};

template<typename> struct rank : public index_const<3> {};

template<typename ...> struct extents : public index_sequence<> {};


template<typename ...T> struct iform_list { typedef index_sequence<iform<T>::value...> type; };

template<typename ...T> using iform_list_t= typename iform_list<T...>::type;

template<typename ...> struct value_type;

template<typename ...T> using value_type_t=typename value_type<T...>::type;

template<typename T> struct value_type<T> { typedef T type; };
template<typename T> struct value_type<T &> { typedef T &type; };
template<typename T> struct value_type<T const &> { typedef T const &type; };

template<typename T> struct value_type<T *> { typedef T type; };
template<typename T, size_type N> struct value_type<T[N]> { typedef T type; };
template<typename T> struct value_type<T const *> { typedef T type; };
template<typename T> struct value_type<T const[]> { typedef T type; };

template<typename T> struct sub_type { typedef T type; };
template<typename T> using sub_type_t = typename sub_type<T>::type;

template<typename ...> struct pod_type;
template<typename ...T> using pod_type_t = typename pod_type<T...>::type;
template<typename T> struct pod_type<T> { typedef T type; };

template<typename T> struct scalar_type { typedef Real type; };

template<typename T> using scalar_type_t=typename scalar_type<T>::type;

template<typename ...> struct is_complex : public std::integral_constant<bool, false> {};

template<typename T> struct is_complex<std::complex<T>> : public std::integral_constant<bool, true> {};


template<typename ...> struct is_scalar : public std::integral_constant<bool, false> {};

template<typename T>
struct is_scalar<T> : public std::integral_constant<bool,
        std::is_arithmetic<std::decay_t<T>>::value || is_complex<std::decay_t<T>>::value>
{
};
template<typename First, typename  ...Others>
struct is_scalar<First, Others...> : public std::integral_constant<bool,
        is_scalar<First>::value && is_scalar<Others...>::value>
{
};

template<typename ...> struct is_field;

template<typename T> struct is_field<T> : public std::integral_constant<bool, false> {};
template<typename First, typename  ...Others>
struct is_field<First, Others...> : public std::integral_constant<bool,
        is_field<First>::value || is_field<Others...>::value>
{
};
template<typename ...> struct is_array;

template<typename T> struct is_array<T> : public std::integral_constant<bool, false> {};

template<typename First, typename  ...Others>
struct is_array<First, Others...> : public std::integral_constant<bool,
        (is_array<First>::value && !is_field<First>::value) || is_array<Others...>::value>
{
};

template<typename ...> struct is_nTuple;

template<typename T> struct is_nTuple<T> : public std::integral_constant<bool, false> {};

template<typename First, typename  ...Others>
struct is_nTuple<First, Others...> : public std::integral_constant<bool,
        (is_nTuple<First>::value && !(is_field<First>::value || is_array<First>::value)) || is_nTuple<Others...>::value>
{
};
template<typename ...> struct is_expression;

template<typename T> struct is_expression<T> : public std::integral_constant<bool, false> {};

template<typename ... T>
struct is_expression<declare::Expression<T...>> : public std::integral_constant<bool, true> {};


template<typename T> struct reference { typedef T type; };
template<typename T> using reference_t=typename reference<T>::type;
template<typename T, int N> struct reference<T[N]> { typedef T *type; };
template<typename T, int N> struct reference<const T[N]> { typedef T const *type; };



//***********************************************************************************************************************


} //namespace traits



} //namespace algebra


template<typename T, size_type ...N> using nTuple=algebra::declare::nTuple_<T, N...>;

template<typename T, size_type N> using Vector=algebra::declare::nTuple_<T, N>;

template<typename T, size_type M, size_type N> using Matrix=algebra::declare::nTuple_<T, M, N>;

template<typename T, size_type ...N> using Tensor=algebra::declare::nTuple_<T, N...>;

template<typename T, size_type N, bool is_slow_first = true> using Array=algebra::declare::Array_<T, N, is_slow_first>;

} //namespace simpla
#endif //SIMPLA_ALGEBRACOMMON_H
