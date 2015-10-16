/**
 * @file parallel_traits.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_PARALLEL_TRAITS_H
#define SIMPLA_PARALLEL_TRAITS_H
namespace simpla
{

namespace parallel
{

template<typename ...T> bool is_ready(T &&...) { return true; }

template<typename ...T> void wait(T &&...) { }

template<typename ...T> void sync(T &&...) { }
}

//namespace parallel
}//namespace simpla

#endif //SIMPLA_PARALLEL_TRAITS_H
