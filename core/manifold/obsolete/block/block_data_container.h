/**
 * @file block_data_container.h
 * @author salmon
 * @date 2015-07-23.
 */

#ifndef SIMPLA_BLOCK_DATA_CONTAINER_H
#define SIMPLA_BLOCK_DATA_CONTAINER_H

#include <map>
#include <memory>

namespace simpla
{
namespace mesh
{
template<typename T> using data_container= std::map<size_t, std::shared_ptr<T> >;
}
}// namespace simpla
#endif //SIMPLA_BLOCK_DATA_CONTAINER_H
