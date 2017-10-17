/**
 * @file type_traits.h
 *
 *  created on: 2014-6-15
 *      Author: salmon
 */

#ifndef SP_TYPE_TRAITS_H_
#define SP_TYPE_TRAITS_H_

#include <initializer_list>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include "host_define.h"

namespace simpla {

#define ENABLE_IF(_COND_) std::enable_if_t<_COND_, void>* _p = nullptr

// CHECK_OPERATOR(is_indexable, [])

typedef std::nullptr_t NullType;

struct EmptyType {};

namespace tags {
struct do_nothing {
    template <typename... Args>
    void operator()(Args&&...) const {}
};
}

namespace traits {

namespace detail {

template <typename _TFun, typename... _Args>
struct check_invocable {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> decltype(std::declval<_U>()(std::declval<_Args>()...));

    template <typename>
    static no test(...);

   public:
    typedef decltype(test<_TFun>(0)) type;

    static constexpr bool value = !std::is_same<type, no>::value;
};

template <typename _TFun, typename _Arg>
struct check_indexable {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> decltype(std::declval<_U>()[std::declval<_Arg>()]);

    template <typename>
    static no test(...);

   public:
    typedef decltype(test<_TFun>(0)) type;

    static constexpr bool value = !std::is_same<type, no>::value;
};
}

template <typename TFun, typename... Args>
struct invoke_result {
    typedef typename detail::check_invocable<TFun, Args...>::type type;
};

template <typename TFun, typename... Args>
using invoke_result_t = typename invoke_result<TFun, Args...>::type;

template <typename TFun, typename... Args>
struct is_invocable
    : public std::integral_constant<
          bool, !std::is_same<typename detail::check_invocable<TFun, Args...>::type, std::false_type>::value> {};

template <typename R, typename TFun, typename... Args>
struct is_invocable_r
    : public std::integral_constant<bool,
                                    std::is_same<typename detail::check_invocable<TFun, Args...>::type, R>::value> {};

template <typename U, typename ArgsTypelist, typename Enable = void>
struct InvokeHelper_ {
    template <typename V, typename... Args>
    static decltype(auto) eval(V& v, Args&&... args) {
        return v;
    }
};

template <typename U, typename... Args>
struct InvokeHelper_<U, std::tuple<Args...>, std::enable_if_t<is_invocable<U, Args...>::value>> {
    template <typename V, typename... Args2>
    static decltype(auto) eval(V& v, Args2&&... args) {
        return v(std::forward<Args>(args)...);
    }
};
template <typename U, typename... Args>
struct InvokeHelper_<const U, std::tuple<Args...>, std::enable_if_t<is_invocable<const U, Args...>::value>> {
    template <typename V, typename... Args2>
    static decltype(auto) eval(V const& v, Args2&&... args) {
        return v(std::forward<Args>(args)...);
    }
};
template <typename U, typename... Args>
decltype(auto) invoke(U& v, Args&&... args) {
    return InvokeHelper_<U, std::tuple<Args...>>::eval(v, std::forward<Args>(args)...);
}
template <typename U, typename... Args>
decltype(auto) invoke(U const& v, Args&&... args) {
    return InvokeHelper_<U, std::tuple<Args...>>::eval(v, std::forward<Args>(args)...);
}
//**********************************************************************************************************************
/**
* @ref http://en.cppreference.com/w/cpp/types/remove_extent
* If T is '''is_indexable''' by '''S''', provides the member typedef type equal to
* decltyp(T[S])
* otherwise type is T. Note that if T is a multidimensional array, only the first dimension is
* removed.
*/

template <typename T, typename TI = int>
struct is_indexable : public std::integral_constant<
                          bool, !std::is_same<typename detail::check_indexable<T, TI>::type, std::false_type>::value> {
};

template <typename T, typename TI = int>
struct index_result {
    typedef typename detail::check_indexable<T, TI>::type type;
};
template <typename U, typename I0, typename Enable = void>
struct IndexHelper_ {
    template <typename V>
    static decltype(auto) eval(V& v, I0 const& s) {
        return v;
    }
};

template <typename U, typename I0>
struct IndexHelper_<U, I0, std::enable_if_t<(is_indexable<std::remove_cv_t<U>, I0>::value)>> {
    template <typename V>
    static decltype(auto) eval(V& v, I0 const& s) {
        return v[s];
    }
};
template <typename U>
decltype(auto) index(U& v) {
    return v;
}
template <typename U, typename I0>
decltype(auto) index(U& v, I0 const& s) {
    return IndexHelper_<U, I0>::eval(v, s);
}
template <typename U, typename I0>
decltype(auto) index(U const& v, I0 const& s) {
    return IndexHelper_<const U, I0>::eval(v, s);
}
template <typename U, typename I0, typename... Others>
decltype(auto) index(U& v, I0 const& s, Others&&... others) {
    return index(index(v, s), std::forward<Others>(others)...);
}

template <typename U>
decltype(auto) recursive_index(U& v, int s, ENABLE_IF((std::rank<U>::value == 0))) {
    return v;
}
template <typename U>
decltype(auto) recursive_index(U const& v, int s, ENABLE_IF((std::rank<U>::value == 0))) {
    return v;
}
template <typename U>
decltype(auto) recursive_index(U& v, int s, ENABLE_IF((std::rank<U>::value > 0))) {
    return recursive_index(v[s % std::extent<std::remove_cv_t<U>>::value], s / std::extent<std::remove_cv_t<U>>::value);
}
template <typename U>
decltype(auto) recursive_index(U const& v, int s, ENABLE_IF((std::rank<U>::value > 0))) {
    return recursive_index(v[s % std::extent<std::remove_cv_t<U>>::value], s / std::extent<std::remove_cv_t<U>>::value);
}

////////////////////////////////////////////////////////////////////////
///// Property queries of n-dimensional array
////////////////////////////////////////////////////////////////////////

template <typename T>
struct reference {
    typedef T type;
};

template <typename T>
struct reference<T&&> {
    typedef T type;
};
template <typename T>
using reference_t = typename reference<T>::type;

template <typename T>
struct value_type {
    typedef T type;
};
template <typename T>
using value_type_t = typename value_type<T>::type;

template <typename T>
struct scalar_type {
    typedef double type;
};
template <typename T>
using scalar_type_t = typename scalar_type<T>::type;

template <typename TExpr>
struct dimension : public std::integral_constant<int, 0> {};

template <typename U>
struct extents : public std::integer_sequence<int, 1> {};

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

    typedef std::conditional_t<std::is_same<_type, no>::value, _T, typename remove_all_extents<_type, _Args>::type>
        type;
};

template <typename _Args>
struct remove_all_extents<std::false_type, _Args> {
    typedef std::false_type type;
};
template <typename T, typename Idx = int>
using remove_all_extents_t = typename remove_all_extents<T, Idx>::type;

template <typename TV, int... I>
struct add_extents;

template <typename T, int... I>
using add_extents_t = typename add_extents<T, I...>::type;

template <typename TV>
struct add_extents<TV> {
    typedef TV type;
};

template <typename TV, int I0, int... I>
struct add_extents<TV, I0, I...> {
    typedef add_extents_t<TV, I...> type[I0];
};

template <typename V, int N>
struct nested_initializer_list;

template <typename V, int N>
using nested_initializer_list_t = typename nested_initializer_list<V, N>::type;

template <typename V>
struct nested_initializer_list<V, 0> {
    typedef V type;
};

template <typename V>
struct nested_initializer_list<V, 1> {
    typedef std::initializer_list<V> type;
};

template <typename V, int N>
struct nested_initializer_list {
    typedef std::initializer_list<nested_initializer_list_t<V, N - 1>> type;
};

template <typename U>
struct nested_initializer_list_traits {
    static constexpr int number_of_levels = 0;
    static void GetDims(U const& list, int* dims) {}
};
template <typename U>
struct nested_initializer_list_traits<std::initializer_list<U>> {
    static constexpr int number_of_levels = nested_initializer_list_traits<U>::number_of_levels + 1;
    static void GetDims(std::initializer_list<U> const& list, std::size_t* dims) { dims[0] = list.size(); }
};
template <typename U>
struct nested_initializer_list_traits<std::initializer_list<std::initializer_list<U>>> {
    static constexpr int number_of_levels = nested_initializer_list_traits<U>::number_of_levels + 2;
    static void GetDims(std::initializer_list<std::initializer_list<U>> const& list, std::size_t* dims) {
        dims[0] = list.size();
        std::size_t max_length = 0;
        for (auto const& item : list) { max_length = (max_length < item.size()) ? item.size() : max_length; }
        dims[1] = max_length;
    }
};

template <int... I>
struct assign_nested_initializer_list;

template <>
struct assign_nested_initializer_list<> {
    template <typename U, typename TR>
    __host__ __device__ static void apply(U& u, TR const& rhs) {
        u = rhs;
    }
};

template <int I0, int... I>
struct assign_nested_initializer_list<I0, I...> {
    template <typename U, typename TR>
    __host__ __device__ static void apply(U& u, std::initializer_list<TR> const& rhs) {
        static_assert(is_indexable<U, int>::value, " illegal value_type_info");

        auto it = rhs.begin();
        auto ie = rhs.end();

        for (int i = 0; i < I0 && it != ie; ++i, ++it) { assign_nested_initializer_list<I...>::apply(u[i], *it); }
    }
};

namespace _detail {
template <bool F>
struct remove_entent_v;
}  // namespace _detail

template <typename T, typename I0>
remove_all_extents_t<T, I0>& get_v(T& v, I0 const* s) {
    return _detail::remove_entent_v<is_indexable<T, I0>::value>::get(v, s);
};

template <typename T, typename I0, typename... Idx>
remove_extents_t<T, I0, Idx...>& get_v(T& v, I0 const& s0, Idx&&... idx) {
    return _detail::remove_entent_v<is_indexable<T, I0>::value>::get(v, s0, std::forward<Idx>(idx)...);
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

namespace _impl {

template <int N>
struct unpack_args_helper {
    template <typename... Args>
    auto eval(Args&&... args) {
        return ((unpack_args_helper<N - 1>(std::forward<Args>(args)...)));
    }
};

template <>
struct unpack_args_helper<0> {
    template <typename First, typename... Args>
    auto eval(First&& first, Args&&... args) {
        return ((std::forward<First>(first)));
    }
};
}  // namespace _impl

template <int N, typename... Args>
auto unpack_args(Args&&... args) {
    return ((_impl::unpack_args_helper<N>(std::forward<Args>(args)...)));
}

template <typename... T>
struct type_list {};

template <size_t, typename...>
struct type_list_N;

template <size_t N>
struct type_list_N<N> {
    typedef std::nullptr_t type;
};

template <typename First, typename... Others>
struct type_list_N<0, First, Others...> {
    typedef First type;
};

template <size_t N, typename First, typename... Others>
struct type_list_N<N, First, Others...> {
    typedef typename type_list_N<N - 1, Others...>::type type;
};

template <size_t N, typename... T>
using type_list_N_t = typename type_list_N<N, T...>::type;

template <typename...>
struct check_type_in_list;
template <>
struct check_type_in_list<> : public std::false_type {};

template <typename T>
struct check_type_in_list<T> : public std::false_type {};
template <typename T>
struct check_type_in_list<T, T> : public std::true_type {};

template <typename L, typename R>
struct check_type_in_list<L, R>
    : public std::integral_constant<
          bool, std::is_same<L, R>::value || std::is_base_of<R, L>::value || std::is_base_of<R, L>::value> {};
template <typename L, typename R, typename... Others>
struct check_type_in_list<L, R, Others...>
    : public std::integral_constant<bool, check_type_in_list<L, R>::value || check_type_in_list<L, Others...>::value> {
};
template <typename L, typename... Others>
struct check_type_in_list<L, type_list<Others...>>
    : public std::integral_constant<bool, check_type_in_list<L, Others...>::value> {};

template <typename TI, typename TJ>
size_type get_value(std::integer_sequence<TI> const& _, TJ* d) {
    return 0;
};

template <typename TI, TI N0, TI... N, typename TJ>
size_type get_value(std::integer_sequence<TI, N0, N...> const& _, TJ* d) {
    if (d != nullptr) {
        d[0] = N0;
        get_value(std::integer_sequence<TI, N...>(), d + 1);
    }
    return sizeof...(N);
};

}  // namespace traits

template <typename Arg0>
static constexpr auto max(Arg0 const& arg0) {
    return arg0;
};
template <typename Arg0, typename... Args>
static constexpr auto max(Arg0 const& arg0, Args&&... args) {
    return arg0 > max(args...) ? arg0 : max(args...);
};

template <typename Arg0>
static constexpr auto min(Arg0 const& arg0) {
    return arg0;
};
template <typename Arg0, typename... Args>
static constexpr auto min(Arg0 const& arg0, Args&&... args) {
    return arg0 < max(args...) ? arg0 : max(args...);
};

template <typename T>
T power2(T const& v) {
    return v * v;
}

template <typename T>
T power3(T const& v) {
    return v * v * v;
}

/**
* @ingroup concept
*  @addtogroup concept_check Concept Checking
*  @{
*/

/**
 * @brief check if T has type member  T::_NAME_
 *  if T has member type _NAME_, provides the member typedef type which is the same as T::_NAME_
 *  else  provides void
 *
  EXAMPLE:

 @code{.cpp}
    ->append/concept/CheckConcept.h>
    #include <iostream>
    using namespace simpla;

    struct Foo { typedef double value_type;};
    struct Goo { typedef int value_type;};
    struct Koo {};

    CHECK_MEMBER_TYPE(value_type, value_type)

    int main(int argc, char** argv) {
        std::cout << " Foo::value_type " << (has_value_type<Foo>::value ? "is" : "is not") <<
 "defined." << std::endl;
        std::cout << " Goo::value_type " << (has_value_type<Goo>::value ? "is" : "is not") <<
 "defined." << std::endl;
        std::cout << " Koo::value_type " << (has_value_type<Koo>::value ? "is" : "is not") <<
 "defined." << std::endl;

        std::cout << " Foo::value_type " << (std::is_same<value_type_t<Foo>, double>::value ? "is" :
 "is not") << " double" << std::endl;
        std::cout << " Goo::value_type " << (std::is_same<value_type_t<Goo>, double>::value ? "is" :
 "is not") << " double" << std::endl;
        std::cout << " Koo::value_type " << (std::is_same<value_type_t<Koo>, double>::value ? "is" :
 "is not") << " double" << std::endl;
    }
 @endcode

 OUTPUT:
  @verbatim
     Foo::value_type is defined
     Goo::value_type is defined
     Koo::value_type is not defined
     Foo::value_type is double
     Goo::value_type is not double
     Koo::value_type is not double
  @endverbatim

 *
 */

#define CHOICE_TYPE_WITH_TYPE_MEMBER(_CHECKER_NAME_, _TYPE_NAME_, _DEFAULT_TYPE_)                           \
    template <typename _T>                                                                                  \
    struct _CHECKER_NAME_ {                                                                                 \
       private:                                                                                             \
        typedef std::true_type yes;                                                                         \
        typedef std::false_type no;                                                                         \
                                                                                                            \
        template <typename _U>                                                                              \
        static auto test(int) -> typename _U::_TYPE_NAME_;                                                  \
        template <typename>                                                                                 \
        static no test(...);                                                                                \
        typedef decltype(test<_T>(0)) result_type;                                                          \
                                                                                                            \
       public:                                                                                              \
        typedef std::conditional_t<std::is_same<result_type, no>::value, _DEFAULT_TYPE_, result_type> type; \
    };                                                                                                      \
    template <typename _T>                                                                                  \
    using _CHECKER_NAME_##_t = typename _CHECKER_NAME_<_T>::type;                                           \
    template <typename _T>                                                                                  \
    struct has_##_CHECKER_NAME_                                                                             \
        : public std::integral_constant<bool,                                                               \
                                        !std::is_same<typename _CHECKER_NAME_<_T>::type, _DEFAULT_TYPE_>::value> {};

#define CHECK_MEMBER_TYPE(_CHECKER_NAME_, _TYPE_NAME_) CHOICE_TYPE_WITH_TYPE_MEMBER(_CHECKER_NAME_, _TYPE_NAME_, void)

/**
 * @brief  check if   type T has a  type member as std::true_type or std::false_type
 *
EXAMPLE:
 @code{.cpp}
    ->append/concept/CheckMemberType.h>
    #include <iostream>
    using namespace simpla;

    struct Foo { typedef std::true_type is_foo;};
    struct Goo { typedef std::false_type is_foo; };
    struct Koo {};

    CHECK_BOOLEAN_TYPE_MEMBER(is_foo, is_foo)

    int main(int argc, char** argv) {
        std::cout << " Foo  " << (is_foo<Foo>::value ? "is" : "is not") << " foo." << std::endl;
        std::cout << " Goo  " << (is_foo<Goo>::value ? "is" : "is not") << " foo." << std::endl;
        std::cout << " Koo  " << (is_foo<Koo>::value ? "is" : "is not") << " foo." << std::endl;
    }
 @endcode

 OUTPUT:

 @verbatim
  Foo  is foo.
  Goo  is not foo.
  Koo  is not foo.

 @endverbatim

 */

#define CHECK_BOOLEAN_TYPE_MEMBER(_CHECKER_NAME_, _MEM_NAME_)                                                     \
    namespace detail {                                                                                            \
    template <typename T>                                                                                         \
    struct _CHECKER_NAME_ {                                                                                       \
       private:                                                                                                   \
        typedef std::true_type yes;                                                                               \
        typedef std::false_type no;                                                                               \
                                                                                                                  \
        template <typename U>                                                                                     \
        static auto test(int) -> typename U::_MEM_NAME_;                                                          \
        template <typename>                                                                                       \
        static no test(...);                                                                                      \
        typedef decltype(test<T>(0)) result_type;                                                                 \
                                                                                                                  \
       public:                                                                                                    \
        typedef std::conditional_t<std::is_same<result_type, yes>::value || std::is_same<result_type, no>::value, \
                                   result_type, no>                                                               \
            type;                                                                                                 \
    };                                                                                                            \
    }                                                                                                             \
    template <typename...>                                                                                        \
    struct _CHECKER_NAME_;                                                                                        \
    template <typename T>                                                                                         \
    struct _CHECKER_NAME_<T> : public detail::_CHECKER_NAME_<T>::type {};

/**
 * @brief check if a type T has member variable _NAME_
 *
 * EXAMPLE:
 * @code{.cpp}
    struct Foo { static constexpr int iform = 2;};
    struct Goo { static constexpr double iform = 2.1; };
    struct Koo {};
    CHECK_MEMBER_STATIC_CONSTEXPR_DATA(has_iform, iform)
    TEST(CheckConceptTest, CheckMemberStaticConstexprData) {
    EXPECT_TRUE(has_iform<Foo>::value);
    EXPECT_TRUE(has_iform<Goo>::value);
    EXPECT_FALSE(has_iform<Koo>::value);
}
 * @endcode
 */

#define CHECK_STATIC_CONSTEXPR_DATA_MEMBER(_CHECKER_NAME_, _V_NAME_)                              \
    namespace detail {                                                                            \
    template <typename _T>                                                                        \
    struct _CHECKER_NAME_ {                                                                       \
       private:                                                                                   \
        typedef std::true_type yes;                                                               \
        typedef std::false_type no;                                                               \
                                                                                                  \
        template <typename U>                                                                     \
        static auto test(int) -> decltype(U::_V_NAME_);                                           \
        template <typename>                                                                       \
        static no test(...);                                                                      \
                                                                                                  \
       public:                                                                                    \
        typedef std::conditional_t<std::is_same<decltype(test<_T>(0)), no>::value, no, yes> type; \
    };                                                                                            \
    }                                                                                             \
    template <typename T>                                                                         \
    struct _CHECKER_NAME_ : public detail::_CHECKER_NAME_<T>::type {};

/**
 * @brief check if T has static  data member  T::_NAME_
     @code{.cpp}
    struct Foo { static constexpr int iform = 2;};
    struct Goo { static constexpr double iform = 2.1; };
    struct Koo {};
    CHECK_MEMBER_STATIC_CONSTEXPR_DATA_VALUE(iform_value, iform, 12)
    TEST(CheckConceptTest, CheckMemberStaticConstexprDataValue) {
    EXPECT_EQ(iform_value<Foo>::value, 2);
    EXPECT_DOUBLE_EQ(iform_value<Goo>::value, 2.1);
    EXPECT_NE(iform_value<Goo>::value, 1);
    EXPECT_EQ(iform_value<Koo>::value, 12);
   }
 * @endcode
 */

#define CHECK_VALUE_OF_STATIC_CONSTEXPR_DATA_MEMBER(_CHECKER_NAME_, _V_NAME_, _DEFAULT_VALUE_) \
    namespace detail {                                                                         \
    CHECK_STATIC_CONSTEXPR_DATA_MEMBER(check_##_CHECKER_NAME_, _V_NAME_)                       \
    template <typename T, bool ENABLE>                                                         \
    struct _CHECKER_NAME_;                                                                     \
    template <typename T>                                                                      \
    struct _CHECKER_NAME_<T, true> {                                                           \
        static constexpr decltype(T::_V_NAME_) value = T::_V_NAME_;                            \
    };                                                                                         \
    template <typename T>                                                                      \
    struct _CHECKER_NAME_<T, false> {                                                          \
        static constexpr decltype(_DEFAULT_VALUE_) value = _DEFAULT_VALUE_;                    \
    };                                                                                         \
    template <typename T>                                                                      \
    constexpr decltype(T::_V_NAME_) _CHECKER_NAME_<T, true>::value;                            \
    template <typename T>                                                                      \
    constexpr decltype(_DEFAULT_VALUE_) _CHECKER_NAME_<T, false>::value;                       \
    } /*namespace detail*/                                                                     \
    template <typename T>                                                                      \
    struct _CHECKER_NAME_ : public detail::_CHECKER_NAME_<T, detail::check_##_CHECKER_NAME_<T>::value> {};

#define CHECK_STATIC_INTEGRAL_CONSTEXPR_DATA_MEMBER(_CHECKER_NAME_, _V_NAME_, _DEFAULT_VALUE_) \
    namespace detail {                                                                         \
    CHECK_STATIC_CONSTEXPR_DATA_MEMBER(has_##_CHECKER_NAME_, _V_NAME_)                         \
    template <typename T, bool ENABLE>                                                         \
    struct _CHECKER_NAME_;                                                                     \
    template <typename T>                                                                      \
    struct _CHECKER_NAME_<T, true> {                                                           \
        static constexpr int value = T::_V_NAME_;                                              \
    };                                                                                         \
    template <typename T>                                                                      \
    struct _CHECKER_NAME_<T, false> {                                                          \
        static constexpr int value = _DEFAULT_VALUE_;                                          \
    };                                                                                         \
    template <typename T>                                                                      \
    constexpr int _CHECKER_NAME_<T, true>::value;                                              \
    template <typename T>                                                                      \
    constexpr int _CHECKER_NAME_<T, false>::value;                                             \
    } /*namespace detail*/                                                                     \
    template <typename T>                                                                      \
    struct _CHECKER_NAME_                                                                      \
        : public std::integral_constant<int,                                                   \
                                        detail::_CHECKER_NAME_<T, detail::has_##_CHECKER_NAME_<T>::value>::value> {};
/**
 * @brief check if T has data memeber _V_NAME_
 *
 @code{.cpp}
 CHECK_MEMBER_DATA(has_data, data)
 TEST(CheckConceptTest, CheckMemberData) {
     EXPECT_TRUE((has_data<Foo>::value));
     EXPECT_TRUE((has_data<Foo, int>::value));
     EXPECT_FALSE((has_data<Foo, double>::value));
     EXPECT_FALSE((has_data<Koo>::value));
     EXPECT_FALSE((has_data<Koo, double>::value));
 }
 @endcode
 */
#define CHECK_DATA_MEMBER(_F_NAME_, _V_NAME_)                                                                   \
    namespace detail {                                                                                          \
    template <typename _T, typename _D>                                                                         \
    struct _F_NAME_ {                                                                                           \
       private:                                                                                                 \
        typedef std::true_type yes;                                                                             \
        typedef std::false_type no;                                                                             \
                                                                                                                \
        template <typename U>                                                                                   \
        static auto test(int) -> decltype(std::declval<U>()._V_NAME_);                                          \
        template <typename>                                                                                     \
        static no test(...);                                                                                    \
        typedef decltype(test<_T>(0)) check_result;                                                             \
                                                                                                                \
       public:                                                                                                  \
        static constexpr bool value = (!std::is_same<check_result, no>::value) &&                               \
                                      (std::is_same<_D, void>::value || std::is_same<check_result, _D>::value); \
    };                                                                                                          \
    }                                                                                                           \
    template <typename T, typename D = void>                                                                    \
    struct _F_NAME_ : public std::integral_constant<bool, detail::_F_NAME_<T, D>::value> {};

//#define CHECK_FUNCTION(_CHECKER_NAME_, _FUN_NAME_)                                                \
//    namespace detail {                                                                            \
//    template <typename...>                                                                        \
//    struct _CHECKER_NAME_ {                                                                       \
//        static constexpr bool value = false;                                                      \
//    };                                                                                            \
//    template <typename _TRet, typename... _Args>                                                  \
//    struct _CHECKER_NAME_<_TRet(_Args...)> {                                                      \
//       private:                                                                                   \
//        typedef std::true_type yes;                                                               \
//        typedef std::false_type no;                                                               \
//        static auto test(_Args &&... args) -> decltype(_FUN_NAME_(std::forward<_Args>(args)...)); \
//        static no test(...);                                                                      \
//                                                                                                  \
//        typedef decltype(test(std::declval<_Args>()...)) check_result;                            \
//                                                                                                  \
//       public:                                                                                    \
//        static constexpr bool value = std::is_same<check_result, _TRet>::value;                   \
//    };                                                                                            \
//    }                                                                                             \
//    template <typename... _Args>                                                                  \
//    struct _CHECKER_NAME_                                                                         \
//        : public std::integral_constant<bool, detail::_CHECKER_NAME_<_Args...>::value> {};

#define CHOICE_TYPE_WITH_FUNCTION_MEMBER(_CHECKER_NAME_, _FUN_NAME_, _DEFAULT_TYPE_)                         \
    template <typename...>                                                                                   \
    struct _CHECKER_NAME_ {                                                                                  \
        typedef _DEFAULT_TYPE_ type;                                                                         \
    };                                                                                                       \
    template <typename _T, typename _TRet>                                                                   \
    struct _CHECKER_NAME_<_T, _TRet()> {                                                                     \
       private:                                                                                              \
        typedef std::true_type yes;                                                                          \
        typedef std::false_type no;                                                                          \
                                                                                                             \
        template <typename U>                                                                                \
        static auto test(int) ->                                                                             \
            typename std::enable_if<sizeof...(_Args) == 0, decltype(std::declval<U>()._FUN_NAME_())>::type;  \
                                                                                                             \
        template <typename>                                                                                  \
        static no test(...);                                                                                 \
                                                                                                             \
        typedef decltype(test<_T>(0)) check_result;                                                          \
                                                                                                             \
       public:                                                                                               \
        typedef std::conditional_t<std::is_same<check_result, _TRet>::value, _T, _DEFAULT_TYPE_> type;       \
    };                                                                                                       \
    template <typename _T, typename _TRet, typename... _Args>                                                \
    struct _CHECKER_NAME_<_T, _TRet(_Args...)> {                                                             \
       private:                                                                                              \
        typedef std::true_type yes;                                                                          \
        typedef std::false_type no;                                                                          \
                                                                                                             \
        template <typename U>                                                                                \
        static auto test(int) ->                                                                             \
            typename std::enable_if<(sizeof...(_Args) > 0),                                                  \
                                    decltype(std::declval<U>()._FUN_NAME_(std::declval<_Args>()...))>::type; \
                                                                                                             \
        template <typename>                                                                                  \
        static no test(...);                                                                                 \
                                                                                                             \
        typedef decltype(test<_T>(0)) check_result;                                                          \
                                                                                                             \
       public:                                                                                               \
        typedef std::conditional_t<std::is_same<check_result, _TRet>::value, _T, _DEFAULT_TYPE_> type;       \
    };                                                                                                       \
    template <typename... _Args>                                                                             \
    using _CHECKER_NAME_##_t = typename _CHECKER_NAME_<_Args...>::type;

#define CHECK_MEMBER_FUNCTION(_CHECKER_NAME_, _FUN_NAME_)                                                    \
    namespace _detail {                                                                                      \
    template <typename...>                                                                                   \
    struct _CHECKER_NAME_ {                                                                                  \
        static constexpr bool value = false;                                                                 \
    };                                                                                                       \
    template <typename _T, typename _TRet, typename... _Args>                                                \
    struct _CHECKER_NAME_<_T, _TRet, _Args...> {                                                             \
       private:                                                                                              \
        typedef std::true_type yes;                                                                          \
        typedef std::false_type no;                                                                          \
                                                                                                             \
        template <typename U>                                                                                \
        static auto test(int) ->                                                                             \
            typename std::enable_if<sizeof...(_Args) == 0, decltype(std::declval<U>()._FUN_NAME_())>::type;  \
                                                                                                             \
        template <typename U>                                                                                \
        static auto test(int) ->                                                                             \
            typename std::enable_if<(sizeof...(_Args) > 0),                                                  \
                                    decltype(std::declval<U>()._FUN_NAME_(std::declval<_Args>()...))>::type; \
                                                                                                             \
        template <typename>                                                                                  \
        static no test(...);                                                                                 \
                                                                                                             \
        typedef decltype(test<_T>(0)) check_result;                                                          \
                                                                                                             \
       public:                                                                                               \
        static constexpr bool value = std::is_same<decltype(test<_T>(0)), _TRet>::value;                     \
    };                                                                                                       \
    } /* namespace _detail*/                                                                                 \
    template <typename... _Args>                                                                             \
    struct _CHECKER_NAME_ : public std::integral_constant<bool, _detail::_CHECKER_NAME_<_Args...>::value> {};

/**
 * @brief
 */
#define CHECK_STATIC_FUNCTION_MEMBER(_CHECKER_NAME_, _FUN_NAME_)                                                      \
    namespace detail {                                                                                                \
                                                                                                                      \
    template <typename _T, typename _TRet, typename... _Args>                                                         \
    struct _CHECKER_NAME_ {                                                                                           \
       private:                                                                                                       \
        typedef std::true_type yes;                                                                                   \
        typedef std::false_type no;                                                                                   \
                                                                                                                      \
        template <typename U>                                                                                         \
        static auto test(int) -> typename std::enable_if<sizeof...(_Args) == 0, decltype(U::_FUN_NAME_())>::type;     \
                                                                                                                      \
        template <typename U>                                                                                         \
        static auto test(int) ->                                                                                      \
            typename std::enable_if<(sizeof...(_Args) > 0), decltype(U::_FUN_NAME_(std::declval<_Args>()...))>::type; \
                                                                                                                      \
        template <typename>                                                                                           \
        static no test(...);                                                                                          \
                                                                                                                      \
        typedef decltype(test<_T>(0)) check_result;                                                                   \
                                                                                                                      \
       public:                                                                                                        \
        static constexpr bool value = std::is_same<decltype(test<_T>(0)), _TRet>::value;                              \
    };                                                                                                                \
    }                                                                                                                 \
    template <typename... _Args>                                                                                      \
    struct _CHECKER_NAME_ : public std::integral_constant<bool, detail::_CHECKER_NAME_<_Args...>::value> {};

#define CHECK_OPERATOR(_NAME_, _OP_)             \
    namespace detail {                           \
    CHECK_MEMBER_FUNCTION(_NAME_, operator _OP_) \
    }                                            \
    template <typename... T>                     \
    struct _NAME_ : public std::integral_constant<bool, detail::_NAME_<T...>::value> {};

// CHECK_OPERATOR(is_callable, ())

namespace traits {
CHECK_STATIC_FUNCTION_MEMBER(has_fancy_type_name, FancyTypeName)

template <typename T, typename Enable = void>
struct type_name {
    static std::string value() { return typeid(T).name(); }
};
template <typename T>
struct type_name<T, std::enable_if_t<has_fancy_type_name<T, std::string>::value>> {
    static std::string value() { return T::FancyTypeName(); }
};

template <>
struct type_name<double> {
    static std::string value() { return "double"; }
};
template <>
struct type_name<int> {
    static std::string value() { return "int"; }
};
template <>
struct type_name<unsigned int> {
    static std::string value() { return "unsigned int"; }
};
template <>
struct type_name<long> {
    static std::string value() { return "long"; }
};
template <>
struct type_name<unsigned long> {
    static std::string value() { return "unsigned long"; }
};

inline std::string to_string() { return ""; }
template <typename Arg0>
std::string to_string(Arg0 const& arg0) {
    return std::to_string(arg0);
}
template <typename Arg0, typename... Args>
std::string to_string(Arg0 const& arg0, Args&&... args) {
    return to_string(arg0) + "," + to_string(std::forward<Args>(args)...);
}
}  // namespace traits
}  // namespace simpla
#endif /* SP_TYPE_TRAITS_H_ */
