//
// Created by salmon on 16-12-22.
//

#ifndef SIMPLA_ALGEBRACOMMON_H
#define SIMPLA_ALGEBRACOMMON_H

#include <simpla/SIMPLA_config.h>
#include <type_traits>
#include <utility>
#include <simpla/mpl/type_traits.h>
#include "../mpl/integer_sequence.h"

namespace simpla
{
enum { VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3, FIBER = 6 };

namespace algebra
{
namespace traits
{

template<typename> struct iform : public index_const<1> {};

template<typename> struct dof : public index_const<1> {};

template<typename> struct rank : public index_const<3> {};

template<typename ...> struct extents : public index_sequence<> {};


template<typename ...T> struct iform_list { typedef index_sequence<iform<T>::value...> type; };

template<typename ...T> using iform_list_t= typename iform_list<T...>::type;

template<typename ...> struct value_type;
template<typename ...T> using value_type_t=typename value_type<T...>::type;
template<typename T> struct value_type<T> { typedef T type; };
template<typename T> struct value_type<T *> { typedef value_type_t<T> type; };

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

template<typename ...> struct is_nTuple : public std::integral_constant<bool, false> {};

template<typename ...> struct is_field : public std::integral_constant<bool, false> {};


template<typename First, typename  ...Others>
struct is_nTuple<First, Others...> : public std::integral_constant<bool,
        (is_nTuple<First>::value || is_nTuple<Others...>::value) && !is_field<First, Others...>::value>
{
};

template<typename First, typename  ...Others>
struct is_field<First, Others...> : public std::integral_constant<bool,
        is_field<First>::value || is_field<Others...>::value>
{
};

template<typename TV> struct reference { typedef TV type; };
template<typename T> using reference_t=typename reference<T>::type;

template<typename T> struct field_value_type { typedef T type; };
template<typename T> using field_value_t=typename field_value_type<T>::type;



//{
//typedef std::conditional_t<
//        iform<T>::value == VERTEX || iform<T>::value == VOLUME, value_type_t<T>,
//        nTuple_ < value_type_t<T>, rank<T>::value> > type;
//};


//template<typename TOP, typename ...Others> struct is_nTuple<Expression < TOP, Others...> > : public is_nTuple<Others...> {};
//template<typename TOP, typename ...Others> struct is_field<Expression < TOP, Others...> > : public is_field<Others...>{};
//template<typename V, size_type ...I> struct is_nTuple<nTuple_ < V, I...> > : public std::integral_constant<bool, true>{};
//template<typename U, typename M, size_type...I> struct is_field<Field_ < U, M, I...> > : public std::integral_constant<bool, true> {};
//template<typename TV, size_type N0, size_type ...N> struct value_type<nTuple_<TV, N0, N...> > { typedef typename value_type<TV>::type type; };
//template<typename TV, typename TM, size_type I, size_type DOF>
//struct reference<Field_<TV, TM, I, DOF> > { typedef Field_<TV, TM, I, DOF> const &type; };
//
//template<typename TV, typename TM, size_type I, size_type DOF> struct iform<Field_<TV, TM, I, DOF> > : public index_const<I> {};
//
//template<typename TV, typename TM, size_type I, size_type DOF> struct dof<Field_<TV, TM, I, DOF> > : public index_const<DOF> {};
//
//template<typename TV, typename TM, size_type I, size_type DOF> struct rank<Field_<TV, TM, I, DOF> > : public rank<TM> { typedef TV type; };
//
//template<typename TV, typename TM, size_type I, size_type DOF> struct value_type<Field_<TV, TM, I, DOF> > { typedef TV type; };
} //namespace traits


//template<typename V> V &
//get_v(V &v, size_type const *s,
//      ENABLE_IF((traits::is_scalar<typename std::decay<V>::type>::value))) { return v; }


//template<typename V> std::complex<V> const &
//get_value(std::complex<V> const &v, size_type const *s) { return v; }
//
//template<typename V> std::complex<V> &
//get_value(std::complex<V> &v, size_type const *s) { return v; }

//template<typename V> auto
//get_v(V *v, size_type const *s) -> decltype((get_v(v[*s], s + 1))) { return get_v(v[*s], s + 1); }

//template<typename V> auto
//get_v(V const *v, size_type const *s) -> decltype((get_v(v[*s], s + 1))) { return get_v(v[*s], s + 1); }


template<typename _T, typename _Args>
struct remove_all_extent
{


private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template<typename _U>
    static auto test(int) ->
    decltype(std::declval<_U>()[std::declval<_Args>()]);

    template<typename> static no test(...);

public:

    static constexpr bool value =
            (!std::is_same<decltype(test<_T>(0)), no>::value)
            || ((std::is_array<_T>::value)
                && (std::is_integral<_Args>::value));

    typedef decltype(test<_T>(0)) _type;

    typedef std::conditional_t<std::is_same<_type, no>::value, _T, typename remove_all_extent<_type, _Args>::type> type;
};

template<typename _Args>
struct remove_all_extent<std::false_type, _Args>
{
    typedef void type;
};

template<typename V> V &
get_v(V &v, size_type const *s,
      ENABLE_IF((!simpla::traits::is_indexable<std::decay_t<V>, size_type>::value))) { return v; }

template<typename V> typename remove_all_extent<V, size_type>::type &
get_v(V &v, size_type const *s,
      ENABLE_IF((simpla::traits::is_indexable<std::decay_t<V>, size_type>::value)))
{
    return get_v(v[*s], s + 1);
}
//template<typename V> auto
//get_v(V const &v, size_type const *s, ENABLE_IF((simpla::traits::is_indexable<std::decay_t<V>, size_type>::value)))
//-> decltype((get_v(v[*s], s + 1))) { return get_v(v[*s], s + 1); }


} //namespace algebra
} //namespace simpla
#endif //SIMPLA_ALGEBRACOMMON_H
