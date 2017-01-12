/**
 * @file check_concept.h
 *
 *  Created on: 2015-1-12
 *      Author: salmon
 */

#ifndef CORE_GPL_CHECK_CONCEPT_H_
#define CORE_GPL_CHECK_CONCEPT_H_

#include <memory>
#include <type_traits>

namespace simpla {
namespace concept {
/**
 * @ingroup concept
 *  @addtogroup concept_check Concept Checking
 *  @{
 */
/**
 * @brief
 */

#define HAS_MEMBER(_NAME_)                                                             \
    template <typename _T>                                                             \
    struct has_member_##_NAME_ {                                                       \
       private:                                                                        \
        typedef std::true_type yes;                                                    \
        typedef std::false_type no;                                                    \
                                                                                       \
        template <typename U>                                                          \
        static auto test(int) -> decltype(std::declval<U>()._NAME_);                   \
        template <typename>                                                            \
        static no test(...);                                                           \
                                                                                       \
       public:                                                                         \
        static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value; \
    };

/**
 * @brief check the member type T::_NAME_
 *  if T has member type _NAME_, provides the member typedef type which is the same as T::_NAME_
 *  else  provides void
 *
  EXAMPLE:

 @code{.cpp}


    #include <simpla/concept/CheckConcept.h>
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
#define CHECK_MEMBER_TYPE(_FUN_NAME_, _TYPE_NAME_)                                                \
    template <typename _T>                                                                        \
    struct _FUN_NAME_ {                                                                           \
       private:                                                                                   \
        typedef std::true_type yes;                                                               \
        typedef std::false_type no;                                                               \
                                                                                                  \
        template <typename U>                                                                     \
        static auto test(int) -> typename U::_TYPE_NAME_;                                         \
        template <typename>                                                                       \
        static no test(...);                                                                      \
        typedef decltype(test<_T>(0)) result_type;                                                \
                                                                                                  \
       public:                                                                                    \
        typedef std::conditional_t<std::is_same<result_type, no>::value, void, result_type> type; \
    };                                                                                            \
    template <typename _T>                                                                        \
    using _FUN_NAME_##_t = typename _FUN_NAME_<_T>::type;                                         \
    template <typename _T>                                                                        \
    struct has_##_FUN_NAME_                                                                       \
        : public std::integral_constant<                                                          \
              bool, !std::is_same<typename _FUN_NAME_<_T>::type, void>::value> {};

/**
 * @brief  check if a type has a member type as std::true_type or std::false_type
 *
EXAMPLE:
 @code{.cpp}
    #include <simpla/concept/CheckMemberType.h>
    #include <iostream>
    using namespace simpla;

    struct Foo { typedef std::true_type is_foo;};
    struct Goo { typedef std::false_type is_foo; };
    struct Koo {};

    CHECK_MEMBER_TYPE_BOOLEAN(is_foo, is_foo)

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

#define CHECK_MEMBER_TYPE_BOOLEAN(_FUN_NAME_, _MEM_NAME_)                    \
    namespace detail {                                                       \
    template <typename T>                                                    \
    struct _FUN_NAME_ {                                                      \
       private:                                                              \
        typedef std::true_type yes;                                          \
        typedef std::false_type no;                                          \
                                                                             \
        template <typename U>                                                \
        static auto test(int) -> typename U::_MEM_NAME_;                     \
        template <typename>                                                  \
        static no test(...);                                                 \
        typedef decltype(test<T>(0)) result_type;                            \
                                                                             \
       public:                                                               \
        typedef std::conditional_t<std::is_same<result_type, yes>::value ||  \
                                       std::is_same<result_type, no>::value, \
                                   result_type, no>                          \
            type;                                                            \
    };                                                                       \
    }                                                                        \
    template <typename T>                                                    \
    struct _FUN_NAME_ : public detail::_FUN_NAME_<T>::type {};

/**
 * @brief check if a type T has member variable _NAME_
 *
 */

#define CHECK_MEMBER_STATIC_CONSTEXPR_DATA(_FUN_NAME_, _V_NAME_)                                  \
    namespace detail {                                                                            \
    template <typename _T>                                                                        \
    struct has_##_FUN_NAME_ {                                                                     \
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
    struct has_##_FUN_NAME_ : public detail::has_##_FUN_NAME_<T>::type {};

#define CHECK_MEMBER_STATIC_CONSTEXPR_DATA_VALUE(_FUN_NAME_, _V_NAME_, _DEFAULT_VALUE_) \
    namespace detail {                                                                  \
    template <typename T, bool ENABLE>                                                  \
    struct _FUN_NAME_;                                                                  \
    template <typename T>                                                               \
    struct _FUN_NAME_<T, true> {                                                        \
        static constexpr decltype(T::_V_NAME_) value = T::_V_NAME_;                     \
    };                                                                                  \
    template <typename T>                                                               \
    struct _FUN_NAME_<T, false> {                                                       \
        static constexpr decltype(_DEFAULT_VALUE_) value = _DEFAULT_VALUE_;             \
    };                                                                                  \
    template <typename T>                                                               \
    constexpr decltype(T::_V_NAME_) _FUN_NAME_<T, true>::value;                         \
    template <typename T>                                                               \
    constexpr decltype(_DEFAULT_VALUE_) _FUN_NAME_<T, false>::value;                    \
    }                                                                                   \
    template <typename T>                                                               \
    struct _FUN_NAME_ : public detail::_FUN_NAME_<T, has_##_FUN_NAME_<T>::value> {};

#define CHECK_MEMBER_DATA(_NAME_, _V_NAME_)                                                        \
    namespace detail {                                                                             \
    template <typename _T, typename _D>                                                            \
    struct has_##_NAME_ {                                                                          \
       private:                                                                                    \
        typedef std::true_type yes;                                                                \
        typedef std::false_type no;                                                                \
                                                                                                   \
        template <typename U>                                                                      \
        static auto test(int) -> decltype(std::declval<U>()._V_NAME_);                             \
        template <typename>                                                                        \
        static no test(...);                                                                       \
        typedef decltype(test<_T>(0)) check_resut;                                                 \
                                                                                                   \
       public:                                                                                     \
        static constexpr bool value =                                                              \
            (!std::is_same<check_resut, no>::value) &&                                             \
            (std::is_same<_D, void>::value || std::is_same<check_resut, _D>::value);                 \
    };                                                                                             \
    }                                                                                              \
    template <typename T, typename D = void>                                                       \
    struct has_##_NAME_ : public std::integral_constant<bool, detail::has_##_NAME_<T, D>::value> { \
    };

#define HAS_MEMBER_FUNCTION(_NAME_)                                                                \
    template <typename _T, typename... _Args>                                                      \
    struct has_member_function_##_NAME_ {                                                          \
       private:                                                                                    \
        typedef std::true_type yes;                                                                \
        typedef std::false_type no;                                                                \
                                                                                                   \
        template <typename U>                                                                      \
        static auto test(int) ->                                                                   \
            typename std::enable_if<sizeof...(_Args) == 0,                                         \
                                    decltype(std::declval<U>()._NAME_())>::type;                   \
                                                                                                   \
        template <typename U>                                                                      \
        static auto test(int) ->                                                                   \
            typename std::enable_if<(sizeof...(_Args) > 0), decltype(std::declval<U>()._NAME_(     \
                                                                std::declval<_Args>()...))>::type; \
                                                                                                   \
        template <typename>                                                                        \
        static no test(...);                                                                       \
                                                                                                   \
       public:                                                                                     \
        static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value;             \
        typedef std::integral_constant<bool, value> value_type;                                    \
    };

#define HAS_CONST_MEMBER_FUNCTION(_NAME_)                                              \
    template <typename _T, typename... _Args>                                          \
    struct has_const_member_function_##_NAME_ {                                        \
       private:                                                                        \
        typedef std::true_type yes;                                                    \
        typedef std::false_type no;                                                    \
                                                                                       \
        template <typename U>                                                          \
        static auto test(int) ->                                                       \
            typename std::enable_if<sizeof...(_Args) == 0,                             \
                                    decltype(std::declval<const U>()._NAME_())>::type; \
                                                                                       \
        template <typename U>                                                          \
        static auto test(int) -> typename std::enable_if<                              \
            (sizeof...(_Args) > 0),                                                    \
            decltype(std::declval<const U>()._NAME_(std::declval<_Args>()...))>::type; \
                                                                                       \
        template <typename>                                                            \
        static no test(...);                                                           \
                                                                                       \
       public:                                                                         \
        static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value; \
        typedef std::integral_constant<bool, value> value_type;                        \
    };

#define HAS_STATIC_MEMBER_FUNCTION(_NAME_)                                                \
    template <typename _T, typename... _Args>                                             \
    struct has_static_member_function_##_NAME_ {                                          \
       private:                                                                           \
        typedef std::true_type yes;                                                       \
        typedef std::false_type no;                                                       \
                                                                                          \
        template <typename U>                                                             \
        static auto test(int) ->                                                          \
            typename std::enable_if<sizeof...(_Args) == 0, decltype(U::_NAME_())>::type;  \
                                                                                          \
        template <typename U>                                                             \
        static auto test(int) ->                                                          \
            typename std::enable_if<(sizeof...(_Args) > 0),                               \
                                    decltype(U::_NAME_(std::declval<_Args>()...))>::type; \
                                                                                          \
        template <typename>                                                               \
        static no test(...);                                                              \
                                                                                          \
       public:                                                                            \
        static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value;    \
        typedef std::integral_constant<bool, value> value_type;                           \
    };

#define HAS_FUNCTION(_NAME_)                                                           \
    template <typename... _Args>                                                       \
    struct has_function_##_NAME_ {                                                     \
       private:                                                                        \
        typedef std::true_type yes;                                                    \
        typedef std::false_type no;                                                    \
                                                                                       \
        static auto test(int) ->                                                       \
            typename std::enable_if<sizeof...(_Args) == 0, decltype(_NAME_())>::type;  \
                                                                                       \
        static auto test(int) ->                                                       \
            typename std::enable_if<(sizeof...(_Args) > 0),                            \
                                    decltype(_NAME_(std::declval<_Args>()...))>::type; \
                                                                                       \
        template <typename>                                                            \
        static no test(...);                                                           \
                                                                                       \
       public:                                                                         \
        static constexpr bool value = !std::is_same<decltype(test(0)), no>::value;     \
    };

#define HAS_OPERATOR(_NAME_, _OP_)                                                       \
    template <typename _T, typename... _Args>                                            \
    struct has_operator_##_NAME_ {                                                       \
       private:                                                                          \
        typedef std::true_type yes;                                                      \
        typedef std::false_type no;                                                      \
                                                                                         \
        template <typename _U>                                                           \
        static auto test(int) ->                                                         \
            typename std::enable_if<sizeof...(_Args) == 0,                               \
                                    decltype(std::declval<_U>().operator _OP_())>::type; \
                                                                                         \
        template <typename _U>                                                           \
        static auto test(int) -> typename std::enable_if<                                \
            (sizeof...(_Args) > 0),                                                      \
            decltype(std::declval<_U>().operator _OP_(std::declval<_Args>()...))>::type; \
                                                                                         \
        template <typename>                                                              \
        static no test(...);                                                             \
                                                                                         \
       public:                                                                           \
        static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value;   \
    };

#define HAS_TYPE(_NAME_)                                                               \
    template <typename _T>                                                             \
    struct has_type_##_NAME_ {                                                         \
       private:                                                                        \
        typedef std::true_type yes;                                                    \
        typedef std::false_type no;                                                    \
                                                                                       \
        template <typename U>                                                          \
        static auto test(int) -> typename U::_NAME_;                                   \
        template <typename>                                                            \
        static no test(...);                                                           \
                                                                                       \
       public:                                                                         \
        static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value; \
    };

namespace _impl {

// HAS_OPERATOR(sub_script, []);
HAS_MEMBER_FUNCTION(at);

}  // namespace _impl

/***
 *   @brief MACRO CHECK_BOOLEAN ( _MEMBER_, _DEFAULT_VALUE_ )
 *    define
 *     template<typename T>
 *     class check_##_MEMBER_
 *     {
 static constexpr bool value;
 *     };
 *
 *     if static T::_MEMBER_ exists, then value = T::_MEMBER_
 *     else value = _DEFAULT_VALUE_
 */
#define CHECK_BOOLEAN(_MEMBER_, _DEFAULT_VALUE_)                                                 \
    template <typename _T_CHKBOOL_>                                                              \
    struct check_##_MEMBER_ {                                                                    \
       private:                                                                                  \
        HAS_STATIC_MEMBER(_MEMBER_);                                                             \
                                                                                                 \
        template <typename, bool>                                                                \
        struct check_boolean;                                                                    \
                                                                                                 \
        template <typename _U>                                                                   \
        struct check_boolean<_U, true> {                                                         \
            static constexpr bool value = (_U::_MEMBER_);                                        \
        };                                                                                       \
                                                                                                 \
        template <typename _U>                                                                   \
        struct check_boolean<_U, false> {                                                        \
            static constexpr bool value = _DEFAULT_VALUE_;                                       \
        };                                                                                       \
                                                                                                 \
       public:                                                                                   \
        static constexpr bool value =                                                            \
            check_boolean<_T_CHKBOOL_, has_static_member_##_MEMBER_<_T_CHKBOOL_>::value>::value; \
    };

#define CHECK_MEMBER_VALUE(_MEMBER_, _DEFAULT_VALUE_)                                          \
    template <typename _T_CHKBOOL_>                                                            \
    struct check_member_value_##_MEMBER_ {                                                     \
       private:                                                                                \
        HAS_STATIC_MEMBER(_MEMBER_);                                                           \
                                                                                               \
        template <typename, bool>                                                              \
        struct check_value;                                                                    \
                                                                                               \
        template <typename _U>                                                                 \
        struct check_value<_U, true> {                                                         \
            static constexpr auto value = (_U::_MEMBER_);                                      \
        };                                                                                     \
                                                                                               \
        template <typename _U>                                                                 \
        struct check_value<_U, false> {                                                        \
            static constexpr auto value = _DEFAULT_VALUE_;                                     \
        };                                                                                     \
                                                                                               \
       public:                                                                                 \
        static constexpr auto value =                                                          \
            check_value<_T_CHKBOOL_, has_static_member_##_MEMBER_<_T_CHKBOOL_>::value>::value; \
    };

// template<typename _T, typename _Args>
// struct is_indexable
//{
// private:
//    typedef std::true_type yes;
//    typedef std::false_type no;
//
//    template<typename _U>
//    static auto test(int) ->
//    decltype(std::declval<_U>()[std::declval<_Args>()]);
//
//    template<typename> static no test(...);
//
// public:
//
//    static constexpr bool value =
//            (!std::is_same<decltype(test<_T>(0)), no>::value)
//            || ((std::is_array<_T>::value)
//                && (std::is_integral<_Args>::value));
//
//};

/**
 * @}
 */
// namespace traits
//{
// template<typename ...Args> using is_callable = boost::proto::is_callable<Args...>;
//}
template <typename...>
struct is_callable;

template <typename _T, typename... _Args>
struct is_callable<_T(_Args...)> {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> typename std::result_of<_U(_Args...)>::type;

    template <typename>
    static no test(...);

   public:
    typedef decltype(test<_T>(0)) type;
    static constexpr bool value = !std::is_same<type, no>::value;
};

template <typename _T, typename... _Args, typename TRes>
struct is_callable<_T(_Args...), TRes> {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> typename std::result_of<_U(_Args...)>::type;

    template <typename>
    static no test(...);

   public:
    typedef decltype(test<_T>(0)) type;
    static constexpr bool value = std::is_same<type, TRes>::value;
};
// template<typename _T, typename ... _Args, class _R>
// struct is_callable<_T(_Args...), _R>
//{
// private:
//    typedef std::true_type yes;
//    typedef std::false_type no;
//
//    template<typename _U>
//    static auto test(int) -> typename std::result_of<_U(_Args...)>::type;
//
//    template<typename> static no test(...);
//
// public:
//
//    static constexpr bool value = (std::is_same<decltype(test<_T>()), _R>::value);
//
//};

template <typename TFun, typename... Args>
auto try_invoke(TFun const &fun, Args &&... args) ->
    typename std::result_of<TFun(Args &&...)>::type {
    return (fun(std::forward<Args>(args)...));
}

template <typename TFun, typename... Args>
auto try_invoke(TFun const &fun, Args &&... args) ->
    typename std::enable_if<!is_callable<TFun, Args &&...>::value, TFun

                            >::type {
    return fun;
}

template <typename _T>
struct is_iterator {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> decltype(std::declval<_U>().operator*());

    template <typename>
    static no test(...);

   public:
    static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value;
};

template <typename _T>
struct is_shared_ptr {
    static constexpr bool value = false;
};
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> {
    static constexpr bool value = true;
};
template <typename T>
struct is_shared_ptr<const std::shared_ptr<T>> {
    static constexpr bool value = true;
};

template <typename, typename>
struct is_indexable;

// std::enable_if_t<_COND_> *__p = nullptr

#define CHECK_FUNCTION_SIGNATURE(_RET_, _FUN_)                                                \
    typename std::enable_if<std::is_same<_RET_, std::result_of_t<_FUN_>>::value>::type *__p = \
        nullptr
}  // namespace concept
}  // namespace simpla

#endif /* CORE_GPL_CHECK_CONCEPT_H_ */
