/**
 * @file dataset_traits.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_DATASET_TRAITS_H
#define SIMPLA_DATASET_TRAITS_H
namespace simpla
{
namespace traits
{

template<typename ...T> void deploy(T &&...) { }

}//namespace dataset
}//namespace simpla
#endif //SIMPLA_DATASET_TRAITS_H
