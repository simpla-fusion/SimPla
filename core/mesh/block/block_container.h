//
// Created by salmon on 7/23/15.
//

#ifndef SIMPLA_BLOCK_CONTAINER_H
#define SIMPLA_BLOCK_CONTAINER_H

#include "block_layout.h"

namespace simpla {
namespace mesh {

template<typename ...> struct Container;
template<int NDIMS, typename TV>

struct Container<TV, BlockLayout<NDIMS> >
{
    typename BlockLayout<NDIMS>::id_tag_type id_tag_type;
    typedef BlockLayout<NDIMS> block_layout_type;

    typedef TV value_type;
    block_layout_type const &m_layout_;
    std::map<id_tag_type, std::shared_ptr<value_type> > m_data_;


};

} // namespace mesh
} // namespace simpla
#endif //SIMPLA_BLOCK_CONTAINER_H
