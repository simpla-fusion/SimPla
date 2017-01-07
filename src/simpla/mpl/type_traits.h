/**
 * @file type_traits.h
 *
 *  created on: 2014-6-15
 *      Author: salmon
 */

#ifndef SP_TYPE_TRAITS_H_
#define SP_TYPE_TRAITS_H_

#include <simpla/SIMPLA_config.h>
#include <stddef.h>
#include <complex>
#include <map>
#include <tuple>
#include <type_traits>
#include "check_concept.h"
#include "integer_sequence.h"
#include "macro.h"
#include "type_cast.h"

namespace simpla {

typedef std::nullptr_t NullType;

struct EmptyType {};

// template<typename, size_type ...> struct nTuple;

namespace tags {
struct do_nothing {
    template <typename... Args>
    void operator()(Args&&...) const {}
};
}

namespace _impl {
template <typename Func, typename Tup, int... index>
auto invoke_helper(Func&& func, Tup&& tup, index_sequence<index...>) {
    return ((func(std::get<index>(std::forward<Tup>(tup))...)));
}

}  // namespace _impl

template <typename Func, typename Tup>
auto invoke(Func&& func, Tup&& tup) {
    return ((_impl::invoke_helper(
        std::forward<Func>(func), std::forward<Tup>(tup),
        make_index_sequence<std::tuple_size<typename std::decay<Tup>::type>::value>())));
}

namespace traits {
typedef std::integral_constant<bool, true> true_t;
typedef std::integral_constant<bool, false> false_t;

template <typename T, class Enable = void>
struct type_id {
   private:
    HAS_STATIC_MEMBER_FUNCTION(class_name);

    static std::string name_(std::true_type) { return T::class_name(); }

    static std::string name_(std::false_type) { return "unknown"; }

   public:
    static std::string name() {
        return name_(
            std::integral_constant<bool, has_static_member_function_class_name<T>::value>());
    }
};

template <size_type I>
struct type_id<std::integral_constant<size_t, I>> {
    static std::string name() { return "[" + simpla::type_cast<std::string>(I) + "]"; }
};

template <int I>
struct type_id<std::integral_constant<int, I>> {
    static std::string name() {
        return std::string("[") + traits::type_cast<int, std::string>::eval(I) + "]";
    }
};
namespace detail {
// CHECK_STATIC_BOOL_MEMBER(is_self_describing)

template <typename T>
struct check_static_bool_member_is_self_describing {
   private:
    HAS_STATIC_TYPE_MEMBER(is_self_describing)

    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename U>
    static auto test(
        int, std::enable_if_t<has_static_type_member_is_self_describing<U, bool>::value>* = nullptr)
        -> std::integral_constant<bool, U::is_self_describing>;

    template <typename>
    static no test(...);

   public:
    static constexpr bool value = !std::is_same<decltype(test<T>(0)), no>::value;
};
}
template <typename T>
struct type_id<
    T, typename std::enable_if_t<detail::check_static_bool_member_is_self_describing<T>::value>> {
    static std::string name() { return T::name()(); }

    static auto data_type() -> decltype(T::data_type()) { return T::data_type(); }
};

// template<typename T, size_type N>
// struct type_id<T[N], void>
//{
//    static std::string name()
//    {
//        return type_id<T[N]>::name() + "[" + traits::type_cast<int, std::string>::eval(I) +
//        "]";
//    }
//
//    static auto data_type() -> decltype(T::data_type()) { return T::data_type(); }
//};

template <typename T, typename... Others>
struct type_id_list {
    static std::string name() {
        return type_id_list<T>::name() + "," + type_id_list<Others...>::name();
    }
};

template <typename T>
struct type_id_list<T> {
    static std::string name() { return type_id<T>::name(); }
};

#define DEFINE_TYPE_ID_NAME(_NAME_)                   \
    template <>                                       \
    struct type_id<_NAME_> {                          \
        static std::string name() { return #_NAME_; } \
    };

DEFINE_TYPE_ID_NAME(double)

DEFINE_TYPE_ID_NAME(float)

DEFINE_TYPE_ID_NAME(int)

DEFINE_TYPE_ID_NAME(long)

#undef DEFINE_TYPE_ID_NAME

////////////////////////////////////////////////////////////////////////
///// Property queries of n-dimensional array
////////////////////////////////////////////////////////////////////////
template <typename _T, typename _Args = int>
struct is_indexable {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> decltype(std::declval<_U>()[std::declval<_Args>()]);

    template <typename>
    static no test(...);

   public:
    static constexpr bool value =
        (!std::is_same<decltype(test<_T>(0)), no>::value) || (std::is_array<_T>::value);
};

/**
 * @ref http://en.cppreference.com/w/cpp/types/remove_extent
 * If T is '''is_indexable''' by '''S''', provides the member typedef type equal to
 * decltyp(T[S])
 * otherwise type is T. Note that if T is a multidimensional array, only the first dimension is
 * removed.
 */

template <typename T, typename Idx = int>
struct remove_extent {
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> decltype(std::declval<_U>()[std::declval<Idx>()]){};

    template <typename>
    static no test(...){};

   public:
    typedef decltype(test<T>(0)) _type;
    typedef std::conditional_t<std::is_same<_type, no>::value, T, _type> type;
};

template <typename T, typename Idx = int>
using remove_extent_t = typename remove_extent<T, Idx>::type;

template <int N, typename T, typename Idx = int>
struct remove_n_extents;

template <int N, typename T, typename Idx = int>
using remove_n_extents_t = typename remove_n_extents<N, T, Idx>::type;

template <typename T, typename Idx>
struct remove_n_extents<0, T, Idx> {
    typedef T type;
};

template <typename T, typename Idx>
struct remove_n_extents<1, T, Idx> {
    typedef remove_extent_t<T> type;
};

template <int N, typename T, typename Idx>
struct remove_n_extents {
    typedef remove_n_extents_t<N - 1, T, Idx> type;
};

template <typename...>
struct remove_extents;

template <typename T, typename I0>
struct remove_extents<T, I0> {
    typedef remove_extent_t<T, I0> type;
};

template <typename T, typename I0, typename... Others>
struct remove_extents<T, I0, Others...> {
    typedef typename remove_extents<remove_extent_t<T, I0>, Others...>::type type;
};

template <typename... T>
using remove_extents_t = typename remove_extents<T...>::type;

/**
 * @ref http://en.cppreference.com/w/cpp/types/remove_all_extents
 * std::remove_all_extents
 *  If T is a multidimensional array of some type X, provides the member typedef type equal to
 * X, otherwise type is T.
 */
template <typename _T, typename _Args>
struct remove_all_extents {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> decltype(std::declval<_U>()[std::declval<_Args>()]);

    template <typename>
    static no test(...);

   public:
    typedef decltype(test<_T>(0)) _type;

    typedef std::conditional_t<std::is_same<_type, no>::value, _T,
                               typename remove_all_extents<_type, _Args>::type>
        type;
};

template <typename _Args>
struct remove_all_extents<std::false_type, _Args> {
    typedef std::false_type type;
};
template <typename T, typename Idx = int>
using remove_all_extents_t = typename remove_all_extents<T, Idx>::type;

template <typename TV, size_type... I>
struct add_extents;

template <typename T, size_type... I>
using add_extents_t = typename add_extents<T, I...>::type;

template <typename TV>
struct add_extents<TV> {
    typedef TV type;
};

template <typename TV, size_type I0, size_type... I>
struct add_extents<TV, I0, I...> {
    typedef add_extents_t<TV, I...> type[I0];
};

template <typename V, size_type N>
struct nested_initializer_list;

template <typename V, size_type N>
using nested_initializer_list_t = typename nested_initializer_list<V, N>::type;

template <typename V>
struct nested_initializer_list<V, 0> {
    typedef V type;
};

template <typename V>
struct nested_initializer_list<V, 1> {
    typedef std::initializer_list<V> type;
};

template <typename V, size_type N>
struct nested_initializer_list {
    typedef std::initializer_list<nested_initializer_list_t<V, N - 1>> type;
};

template <size_type... I>
struct assign_nested_initializer_list;

template <>
struct assign_nested_initializer_list<> {
    template <typename U, typename TR>
    static inline void apply(U& u, TR const& rhs) {
        u = rhs;
    }
};

template <size_type I0, size_type... I>
struct assign_nested_initializer_list<I0, I...> {
    template <typename U, typename TR>
    static inline void apply(U& u, std::initializer_list<TR> const& rhs) {
        static_assert(is_indexable<U, int>::value, " illegal type");

        auto it = rhs.begin();
        auto ie = rhs.end();

        for (int i = 0; i < I0 && it != ie; ++i, ++it) {
            assign_nested_initializer_list<I...>::apply(u[i], *it);
        }
    }
};

/**
 *  alt. of std::rank
 *  @quto http://en.cppreference.com/w/cpp/types/rank
 *  If T is an array type, provides the member constant
 *  value equal to the number of dimensions of the array.
 *  For any other type, value is 0.
 */
template <typename T, typename Idx = int>
struct rank
    : public index_const<(!is_indexable<T, Idx>::value) ? 0
                                                        : 1 + rank<remove_extent<T, Idx>>::value> {
};

namespace _detail {
template <bool F>
struct remove_entent_v;
}

template <typename T, typename I0>
remove_all_extents_t<T, I0>& get_v(T& v, I0 const* s) {
    return _detail::remove_entent_v<is_indexable<T, I0>::value>::get(v, s);
};

template <typename T, typename I0, typename... Idx>
remove_extents_t<T, I0, Idx...>& get_v(T& v, I0 const& s0, Idx&&... idx) {
    return _detail::remove_entent_v<is_indexable<T, I0>::value>::get(v, s0,
                                                                     std::forward<Idx>(idx)...);
};

namespace _detail {

template <>
struct remove_entent_v<true> {
    template <typename T, typename I0>
    static remove_all_extents_t<T, I0>& get(T& v, I0 const* s) {
        return get_v(v[*s], s + 1);
    };

    template <typename T, typename I0, typename... Idx>
    static remove_extents_t<T, I0, Idx...>& get(T& v, I0 const& s0, Idx&&... idx) {
        return get_v(v[s0], std::forward<Idx>(idx)...);
        ;
    };
};

template <>
struct remove_entent_v<false> {
    template <typename T, typename... Args>
    static T& get(T& v, Args&&...) {
        return v;
    };
};
}  // namespace _detail{

/**
 * alt. of std::extent
 *  @quto http://en.cppreference.com/w/cpp/types/extent
 *  If T is an array type, provides the member constant value equal to
 * the number of elements along the Nth dimension of the array, if N
 * is in [0, std::rank<T>::value). For any other type, or if T is array
 * of unknown bound along its first dimension and N is 0, value is 0.
 */

template <typename T, int N = 0>
struct extent : public std::extent<T, N> {};

template <typename T>
struct size : public std::integral_constant<size_t, 1> {};

template <typename _Tp, _Tp... N>
struct extent<integer_sequence<_Tp, N...>, 0> : public index_const<sizeof...(N)> {};

//**********************************************************************************************************************

// template<typename T>
// struct value_type { typedef typename std::conditional<std::is_scalar<T>::value, T,
// std::nullptr_t>::type type; };
//
// template<typename T> struct value_type<std::complex<T>> { typedef std::complex<T> type; };
//
// template<> struct value_type<std::string> { typedef std::string type; };
//
// template<typename _Tp, _Tp ...N> struct value_type<integer_sequence<_Tp, N...> > { typedef
// _Tp type; };
//
// template<typename T> using value_type_t=typename value_type<T>::type;

template <typename T>
struct key_type {
    typedef int type;
};
template <typename T>
using key_type_t = typename key_type<T>::type;

namespace _impl {

template <int N>
struct unpack_args_helper {
    template <typename... Args>
    auto eval(Args&&... args) AUTO_RETURN((unpack_args_helper<N - 1>(std::forward<Args>(args)...)))
};

template <>
struct unpack_args_helper<0> {
    template <typename First, typename... Args>
    auto eval(First&& first, Args&&... args) AUTO_RETURN((std::forward<First>(first)))
};
}  // namespace _impl

template <int N, typename... Args>
auto unpack_args(Args&&... args) {
    return ((_impl::unpack_args_helper<N>(std::forward<Args>(args)...)));
}

template <typename T0>
auto max(T0 const& first) {
    return ((first));
}

template <typename T0, typename... Others>
auto max(T0 const& first, Others const&... others) {
    return ((std::max(first, max(others...))));
}

template <typename T0>
auto min(T0 const& first) {
    return ((first));
}

template <typename T0, typename... Others>
auto min(T0 const& first, Others const&... others) {
    return ((std::min(first, min(others...))));
}

template <typename T>
auto distance(T const& b, T const& e) {
    return (((e - b)));
}

// template<typename T, typename TI>auto index(std::shared_ptr<T> &v, TI const &s)
// AUTO_RETURN(v.get()[s])
//
// template<typename T, typename TI>auto index(std::shared_ptr<T> const &v, TI const &s)
// AUTO_RETURN(v.get()[s])

namespace _impl {
template <int N>
struct recursive_try_index_aux {
    template <typename T, typename TI>
    static auto eval(T& v, TI const* s) {
        return ((recursive_try_index_aux<N - 1>::eval(v[s[0]], s + 1)));
    }
};

template <>
struct recursive_try_index_aux<0> {
    template <typename T, typename TI>
    static auto eval(T& v, TI const* s) AUTO_RETURN((v))
};
}  // namespace _impl

// template<typename U, typename TIndex> U const &
// index(U const *v, TIndex const &i) { return v[i]; };
//
// template<typename U, typename TIndex> U const &
// index(U const &v, TIndex const &i) { return v; };
//
//
// template<typename U, typename TIndex>
// typename std::remove_extent<U>::type const &
// index(U const &v, TIndex const &i, ENABLE_IF(std::is_array<U>::vaule)) { return v[i]; };
//
// template<typename U, typename TIndex>
// U const &
// index(U const &v, TIndex const &i, ENABLE_IF(std::is_arithmetic<U>::vaule)) { return v; };

// template<typename T, typename TI>
// auto index(T &v, TI s, ENABLE_IF((!is_indexable<T, TI>::value))) AUTO_RETURN((v))
//
// template<typename T, typename TI>
// auto index(T &v, TI s, ENABLE_IF((is_indexable<T, TI>::value))) AUTO_RETURN((v[s]))
//
// template<typename T, typename TI>
// auto index(T &v, TI *s, ENABLE_IF((is_indexable<T, TI>::value)))
// AUTO_RETURN((_impl::recursive_try_index_aux<traits::rank<T>::value>::eval(v, s)))
//
// template<typename T, typename TI, size_type N>
// auto index(T &v, nTuple<TI, N> const &s, ENABLE_IF((is_indexable<T, TI>::value)))
// AUTO_RETURN((_impl::recursive_try_index_aux<N>::eval(v, s)))
//
//
// template<int N, typename T> struct access;
//
// template<int N, typename T>
// struct access
//{
//    static constexpr auto get(T &v) AUTO_RETURN((v))
//
//    template<typename U> static void set(T &v, U const &u) { v = static_cast<T>(u); }
//};
//
// template<int N, typename ...T>
// struct access<N, std::tuple<T...>>
//{
//    static constexpr auto get(std::tuple<T...> &v) AUTO_RETURN((std::get<N>(v)))
//
//    static constexpr auto get(std::tuple<T...> const &v) AUTO_RETURN((std::get<N>(v)))
//
//    template<typename U> static void set(std::tuple<T...> &v, U const &u) { get(v) = u; }
//};
//
// template<int N, typename T>
// struct access<N, T *>
//{
//    static constexpr auto get(T *v) AUTO_RETURN((v[N]))
//
//    static constexpr auto get(T const *v) AUTO_RETURN((v[N]))
//
//    template<typename U> static void set(T *v, U const &u) { get(v) = u; }
//};
//
// template<int N, typename T0, typename T1>
// struct access<N, std::pair<T0, T1>>
//{
//    static constexpr auto get(std::pair<T0, T1> &v) AUTO_RETURN((std::get<N>(v)))
//
//    static constexpr auto get(std::pair<T0, T1> const &v) AUTO_RETURN((std::get<N>(v)))
//
//    template<typename U> static void set(std::pair<T0, T1> &v, U const &u) { get(v) = u; }
//};
////namespace _impl
////{
////
////template<int ...N> struct access_helper;
////
////template<int N0, int ...N>
////struct access_helper<N0, N...>
////{
////
////    template<typename T>
////    static constexpr auto get(T const &v)
/// AUTO_RETURN((access_helper<N...>::get(access_helper<N0>::get((v)))))
////
////    template<typename T>
////    static constexpr auto get(T &v)
/// AUTO_RETURN((access_helper<N...>::get(access_helper<N0>::get((v)))))
////
////    template<typename T, typename U>
////    static void set(T &v, U const &u) { access_helper<N0, N...>::get(v) = u; }
////
////};
////
////template<int N>
////struct access_helper<N>
////{
////    template<typename T> static constexpr auto get(T &v) AUTO_RETURN((access<N, T>::get(v)))
////
////    template<typename T> static constexpr auto get(T const &v) AUTO_RETURN((access<N,
/// T>::get(v)))
////
////    template<typename T, typename U> static void set(T &v, U const &u) { access<N,
/// T>::set(v, u); }
////
////};
////
////template<>
////struct access_helper<>
////{
////    template<typename T> static constexpr T &get(T &v) { return v; }
////
////    template<typename T> static constexpr T const &get(T const &v) { return v; }
////
////    template<typename T, typename U> static void set(T &v, U const &u) { v = u; }
////
////};
////}  // namespace _impl
// template<int N, typename ...T> auto get(std::tuple<T...> &v) AUTO_RETURN((std::get<N>(v)))
//
// template<int ...N, typename T> auto get(T &v)
// AUTO_RETURN((_impl::access_helper<N...>::get(v)))
//
// template<int ...N, typename T> auto get(T const &v)
// AUTO_RETURN((_impl::access_helper<N...>::get(v)))

template <int, typename...>
struct unpack_type;

template <int N>
struct unpack_type<N> {
    typedef std::nullptr_t type;
};

template <typename First, typename... Others>
struct unpack_type<0, First, Others...> {
    typedef First type;
};

template <int N, typename First, typename... Others>
struct unpack_type<N, First, Others...> {
    typedef typename unpack_type<N - 1, Others...>::type type;
};

template <int N, typename... T>
using unpack_t = typename unpack_type<N, T...>::type;

}  // namespace traits

template <typename T>
T power2(T const& v) {
    return v * v;
}

template <typename T>
T power3(T const& v) {
    return v * v * v;
}

template <typename T0>
T0 max(T0 const& first) {
    return first;
};

template <typename T0, typename T1>
T0 max(T0 const& first, T1 const& second) {
    return std::max(first, second);
};

template <typename T0, typename... O>
T0 max(T0 const& first, O&&... others) {
    return max(first, max(std::forward<O>(others)...));
};

template <typename T0>
T0 min(T0 const& first) {
    return first;
};

template <typename T0, typename T1>
T0 min(T0 const& first, T1 const& second) {
    return std::min(first, second);
};

template <typename T0, typename... Others>
T0 min(T0 const& first, Others&&... others) {
    return min(first, min(std::forward<Others>(others)...));
};

}  // namespace simpla
#endif /* SP_TYPE_TRAITS_H_ */
