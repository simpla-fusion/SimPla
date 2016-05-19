/**
 * @file MeshAttribute.cpp
 * @author salmon
 * @date 2016-05-19.
 */

#include "MeshAttribute.h"
#include "../gtl/MemoryPool.h"

void simpla::mesh::AttributeBase::deploy()
{
    int iform = entity_type();

    size_t ele_s = element_size_in_byte();

    for (auto const &item:m_atlas_)
    {
        base_container::emplace(
                std::make_pair(item.uuid(),
                               gtl::sp_alloc_memory(item.number_of_entities_in_memory(iform) * ele_s)));

    }
}