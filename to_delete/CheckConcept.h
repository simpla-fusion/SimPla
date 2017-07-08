/**
 * @file check_concept.h
 *
 *  Created on: 2015-1-12
 *      Author: salmon
 */

#ifndef CORE_CHECK_CONCEPT_H_
#define CORE_CHECK_CONCEPT_H_

#include <type_traits>
#include <utility>  //for std::forward
namespace simpla {
namespace concept {

namespace detail {
template <typename, typename R = void>
struct is_callable;

template <typename _TFun, typename... _Args, typename _R>
struct is_callable<_TFun(_Args...), _R> {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> std::result_of_t<_U(_Args...)>;

    template <typename>
    static no test(...);
    typedef decltype(test<_TFun>(0)) check_result;

   public:
    static constexpr bool value = std::is_same<check_result, _R>::value;
};
template <typename _TFun, typename... _Args, typename _R>
struct is_callable<_TFun(_Args...) const, _R> {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> std::result_of_t<_U(_Args...) const>;

    template <typename>
    static no test(...);
    typedef decltype(test<const _TFun>(0)) check_result;

   public:
    static constexpr bool value = std::is_same<check_result, _R>::value;
};

template <typename _TFun, typename... _Args>
struct is_callable<_TFun(_Args...)> {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> std::result_of_t<_U(_Args...)>;

    template <typename>
    static no test(...);
    typedef decltype(test<_TFun>(0)) check_result;

   public:
    static constexpr bool value = !std::is_same<check_result, no>::value;
};

template <typename _TFun, typename... _Args>
struct is_callable<_TFun(_Args...) const> {
   private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> std::result_of_t<_U(_Args...) const>;

    template <typename>
    static no test(...);
    typedef decltype(test<const _TFun>(0)) check_result;

   public:
    static constexpr bool value = !std::is_same<check_result, no>::value;
};
}

template <typename T, typename R = void>
struct is_callable : public std::integral_constant<bool, detail::is_callable<T, R>::value> {};
template <typename T>
struct is_callable<T> : public std::integral_constant<bool, detail::is_callable<T>::value> {};

template <typename TFun, typename... Args>
auto try_invoke(TFun const &fun, Args &&... args) -> typename std::result_of<TFun(Args &&...)>::type {
    return (fun(std::forward<Args>(args)...));
}

template <typename TFun, typename... Args>
auto try_invoke(TFun const &fun, Args &&... args) ->
    typename std::enable_if<!is_callable<TFun, Args &&...>::value, TFun>::type {
    return fun;
}

}  // namespace concept
}  // namespace simpla

#endif /* CORE_CHECK_CONCEPT_H_ */
