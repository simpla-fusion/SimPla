//
// Created by salmon on 16-5-27.
//

#ifndef SIMPLA_PORT_CXX14_H_H
#define SIMPLA_PORT_CXX14_H_H
#if __cplusplus < 201402L

#include <type_traits>

namespace std
{
///// Alias template for aligned_storage
//template<size_t _Len, size_t _Align =
//__alignof__(typename __aligned_storage_msa<_Len>::__type)>
//using aligned_storage_t = typename aligned_storage<_Len, _Align>::type;
//
//template<size_t _Len, typename... _Types>
//using aligned_union_t = typename aligned_union<_Len, _Types...>::type;

/// Alias template for decay
template<typename _Tp>
using decay_t = typename decay<_Tp>::type;

/// Alias template for enable_if
template<bool _Cond, typename _Tp = void>
using enable_if_t = typename enable_if<_Cond, _Tp>::type;

/// Alias template for conditional
template<bool _Cond, typename _Iftrue, typename _Iffalse>
using conditional_t = typename conditional<_Cond, _Iftrue, _Iffalse>::type;

/// Alias template for common_type
template<typename... _Tp>
using common_type_t = typename common_type<_Tp...>::type;

/// Alias template for underlying_type
template<typename _Tp>
using underlying_type_t = typename underlying_type<_Tp>::type;

/// Alias template for result_of
template<typename _Tp>
using result_of_t = typename result_of<_Tp>::type;
}


#else



#endif
#endif //SIMPLA_PORT_CXX14_H_H
