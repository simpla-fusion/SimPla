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
#include "port_cxx14.h"


namespace simpla
{

namespace traits
{

/**
 * @ingroup toolbox
 *  @addtogroup concept_check Concept Checking
 *  @{
 */
#define HAS_MEMBER(_NAME_)                                                                 \
template<typename _T>                                                                      \
struct has_member_##_NAME_                                                                 \
{                                                                                          \
private:                                                                                   \
    typedef std::true_type yes;                                                            \
    typedef std::false_type no;                                                            \
                                                                                           \
    template<typename U>                                                                   \
    static auto test(int) ->  decltype(std::declval<U>()._NAME_  )   ;                     \
    template<typename > static no test(...);                                               \
                                                                                           \
public:                                                                                    \
    static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;        \
};

#define HAS_TYPE_MEMBER(_NAME_)                                                            \
template<typename _T,typename _D>                                                          \
struct has_type_member_##_NAME_                                                            \
{                                                                                          \
private:                                                                                   \
    typedef std::true_type yes;                                                            \
    typedef std::false_type no;                                                            \
                                                                                           \
    template<typename U>                                                                   \
    static auto test(int) ->  decltype(std::declval<U>()._NAME_  )   ;                     \
    template<typename > static no test(...);                                               \
                                                                                           \
public:                                                                                    \
    static constexpr bool value = std::is_same<decltype(test< _T>(0)), _D>::value;         \
};                                                                                         \
                                                                                           \
template<typename _T, typename _D>                                                         \
typename std::enable_if<has_type_member_##_NAME_<_T, _D>::value, _D>::type                 \
get_member_##_NAME_(_T const & c, _D const & def){    return c._NAME_; }                     \
template<typename _T, typename _D>                                                         \
typename std::enable_if<!has_type_member_##_NAME_<_T, _D>::value, _D>::type                \
get_member_##_NAME_(_T const & c, _D const & def){    return def;}                           \


#define HAS_STATIC_MEMBER(_NAME_)                                                                 \
template<typename _T>                                                                      \
struct has_static_member_##_NAME_                                                                 \
{                                                                                          \
private:                                                                                   \
    typedef std::true_type yes;                                                            \
    typedef std::false_type no;                                                            \
                                                                                           \
    template<typename U>                                                                   \
    static auto test(int) ->  decltype(U::_NAME_  )   ;                     \
    template<typename > static no test(...);                                               \
                                                                                           \
public:                                                                                    \
    static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;        \
};


#define HAS_STATIC_TYPE_MEMBER(_NAME_)                                                            \
template<typename _T,typename _D>                                                          \
struct has_static_type_member_##_NAME_                                                            \
{                                                                                          \
private:                                                                                   \
    typedef std::true_type yes;                                                            \
    typedef std::false_type no;                                                            \
                                                                                           \
    template<typename U>                                                                   \
    static auto test(int) -> decltype( U::_NAME_  ) ;                     \
    template<typename > static no test(...);                                               \
                                                                                           \
public:                                                                                    \
    static constexpr bool value = std::is_same<decltype(test< _T>(0)), _D>::value;         \
};
#define CHECK_STATIC_BOOL_MEMBER(_NAME_)                                                                 \
template<typename _T>                                                                      \
struct check_static_bool_member_##_NAME_                                                                 \
{                                                                                          \
private:                                                                                   \
    typedef std::true_type yes;                                                            \
    typedef std::false_type no;                                                            \
                                                                                           \
    template<typename U>                                                                   \
    static auto test(int) ->  decltype(U::_NAME_  )      ;                                 \
    template<typename > static no test(...);                                               \
                                                                                           \
public:                                                                                    \
    static constexpr bool value =  !std::is_same<decltype(test< _T>(0)), no>::value;        \
};
#define HAS_MEMBER_FUNCTION(_NAME_)                                                                   \
template<typename _T, typename ..._Args>                                                                \
struct has_member_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
    typedef std::true_type yes;                                                                       \
    typedef std::false_type no;                                                                       \
                                                                                                      \
    template<typename U>                                                                              \
    static auto test(int) ->                                                                          \
    typename std::enable_if< sizeof...(_Args)==0,                                                      \
    decltype(std::declval<U>()._NAME_() )>::type;                                                       \
                                                                                                      \
    template<typename U>                                                                              \
    static auto test(int) ->                                                                          \
    typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
    decltype(std::declval<U>()._NAME_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
    template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                       \
    static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;                     \
    typedef std::integral_constant<bool, value> value_type;                                         \
};

#define HAS_CONST_MEMBER_FUNCTION(_NAME_)                                                                   \
template<typename _T, typename ..._Args>                                                                \
struct has_const_member_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
    typedef std::true_type yes;                                                                       \
    typedef std::false_type no;                                                                       \
                                                                                                      \
    template<typename U>                                                                              \
    static auto test(int) ->                                                                          \
    typename std::enable_if< sizeof...(_Args)==0,                                                      \
    decltype(std::declval<const U>()._NAME_() )>::type;                                                       \
                                                                                                      \
    template<typename U>                                                                              \
    static auto test(int) ->                                                                          \
    typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
    decltype(std::declval<const U>()._NAME_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
    template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
    static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;                     \
    typedef std::integral_constant<bool, value> value_type;                                         \
};

#define HAS_STATIC_MEMBER_FUNCTION(_NAME_)                                                                   \
template<typename _T, typename ..._Args>                                                                \
struct has_static_member_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
    typedef std::true_type yes;                                                                       \
    typedef std::false_type no;                                                                       \
                                                                                                      \
    template<typename U>                                                                              \
    static auto test(int) ->                                                                          \
    typename std::enable_if< sizeof...(_Args)==0,                                                      \
    decltype(U::_NAME_() )>::type;                                                       \
                                                                                                      \
    template<typename U>                                                                              \
    static auto test(int) ->                                                                          \
    typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
    decltype(U::_NAME_(std::declval<_Args>()...) )>::type;                                                    \
                                                                                                      \
    template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
      static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;                     \
    typedef std::integral_constant<bool, value> value_type;                                         \
};

#define HAS_FUNCTION(_NAME_)                                                                   \
template< typename ..._Args>                                                                \
struct has_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
    typedef std::true_type yes;                                                                       \
    typedef std::false_type no;                                                                       \
                                                                                                      \
    static auto test(int) ->                                                                          \
    typename std::enable_if< sizeof...(_Args)==0,                                                      \
    decltype(_NAME_() )>::type;                                                       \
                                                                                                      \
    static auto test(int) ->                                                                          \
    typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
    decltype(_NAME_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
    template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
    static constexpr bool value = !std::is_same<decltype(test(0)), no>::value;                     \
};

#define HAS_OPERATOR(_NAME_, _OP_)                                                                   \
template<typename _T, typename ... _Args>                                                                \
struct has_operator_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
    typedef std::true_type yes;                                                                       \
    typedef std::false_type no;                                                                       \
                                                                                                      \
    template<typename _U>                                                                              \
    static auto test(int) ->                                                                          \
    typename std::enable_if< sizeof...(_Args)==0,                                                      \
    decltype(std::declval<_U>().operator _OP_() )>::type;                                                       \
                                                                                                      \
    template<typename _U>                                                                              \
    static auto test(int) ->                                                                          \
    typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
    decltype(std::declval<_U>().operator _OP_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
    template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
    static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value;                     \
};

#define HAS_TYPE(_NAME_)                                                                   \
template<typename _T> struct has_type_##_NAME_                                                     \
{                                                                                                     \
private:                                                                                              \
    typedef std::true_type yes;                                                                       \
    typedef std::false_type no;                                                                       \
                                                                                                      \
    template<typename U> static auto test(int) ->typename U::_NAME_;                                  \
    template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
    static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;                   \
}                                                                                                     \
;

namespace _impl
{

//HAS_OPERATOR(sub_script, []);
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
#define CHECK_BOOLEAN(_MEMBER_, _DEFAULT_VALUE_)                    \
template<typename _T_CHKBOOL_>                                                  \
struct check_##_MEMBER_                                               \
{  private:                                                           \
    HAS_STATIC_MEMBER(_MEMBER_);                                             \
                                                                      \
    template<typename,bool> struct check_boolean;                   \
                                                                      \
    template<typename _U >                                             \
    struct check_boolean<_U,true>                                      \
    {                                                                 \
        static constexpr bool value = ( _U:: _MEMBER_);               \
    };                                                                \
                                                                      \
    template<typename _U >                                             \
    struct check_boolean<_U,false>                                     \
    {                                                                 \
        static constexpr bool value = _DEFAULT_VALUE_;                \
    };                                                                \
  public:                                                             \
    static constexpr bool value =                                     \
           check_boolean<_T_CHKBOOL_,has_static_member_##_MEMBER_<_T_CHKBOOL_>::value>::value;   \
};

#define CHECK_MEMBER_VALUE(_MEMBER_, _DEFAULT_VALUE_)                    \
template<typename _T_CHKBOOL_>                                                  \
struct check_member_value_##_MEMBER_                                               \
{  private:                                                           \
    HAS_STATIC_MEMBER(_MEMBER_);                                             \
                 \
    template<typename,bool> struct check_value;                   \
                                                                      \
    template<typename _U >                                             \
    struct check_value<_U,true>                                      \
    {                                                                 \
        static constexpr auto value = ( _U:: _MEMBER_);               \
    };                                                                \
                                                                      \
    template<typename _U >                                             \
    struct check_value<_U,false>                                     \
    {                                                                 \
        static constexpr auto value = _DEFAULT_VALUE_;                \
    };                                                                \
  public:                                                             \
    static constexpr auto value =                                     \
       check_value<_T_CHKBOOL_,has_static_member_##_MEMBER_<_T_CHKBOOL_>::value>::value;   \
};

template<typename _T, typename _Args>
struct is_indexable
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

};

/**
 * @}
 */
//namespace traits
//{
//template<typename ...Args> using is_callable = boost::proto::is_callable<Args...>;
//}
template<typename ...> struct is_callable;

template<typename _T, typename ... _Args, class _R>
struct is_callable<_T(_Args...), _R>
{
private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template<typename _U>
    static auto test(int) -> typename std::result_of<_U(_Args...)>::type;

    template<typename> static no test(...);

public:

    static constexpr bool value =
            (std::is_same<decltype(test<_T>()), _R>::value);

};

template<typename TFun, typename ... Args>
auto try_invoke(TFun const &fun, Args &&...args) ->
typename std::result_of<TFun(Args &&...)>::type
{
    return (fun(std::forward<Args>(args)...));
}

template<typename TFun, typename ... Args>
auto try_invoke(TFun const &fun, Args &&...args) ->
typename std::enable_if<!is_callable<TFun, Args &&...>::value, TFun

>::type
{
    return
            fun;
}

template<typename _T>
struct is_iterator
{
private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template<typename _U>
    static auto test(int) ->
    decltype(std::declval<_U>().operator*());

    template<typename> static no test(...);

public:

    static constexpr bool value =
            !std::is_same<decltype(test<_T>(0)), no>::value;
};

template<typename _T>
struct is_shared_ptr
{
    static constexpr bool value = false;
};
template<typename T>
struct is_shared_ptr<std::shared_ptr<T>>
{
    static constexpr bool value = true;
};
template<typename T>
struct is_shared_ptr<const std::shared_ptr<T>>
{
    static constexpr bool value = true;
};
template<typename ...> struct is_callable;
template<typename, typename> struct is_indexable;


}  // namespace traits



#define ENABLE_IF(_COND_) typename std::enable_if<_COND_, void>::type *_p = nullptr
 //std::enable_if_t<_COND_> *__p = nullptr

#define CHECK_FUNCTION_SIGNATURE(_RET_, _FUN_) typename std::enable_if<std::is_same<_RET_,std::result_of_t<_FUN_>>::value>::type *__p = nullptr

template<typename ...>
struct is_expression { static constexpr bool value = false; };

}// namespace simpla

#endif /* CORE_GPL_CHECK_CONCEPT_H_ */
