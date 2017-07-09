//
// Created by salmon on 16-10-10.
//

#ifndef SIMPLA_MODELSELECT_H
#define SIMPLA_MODELSELECT_H

#include <simpla/engine/SPObject.h>
#include <functional>
namespace simpla {
namespace mesh {

template <typename TM, typename... Args>
EntityRange select(TM const &m, MeshEntityType iform, Args &&... args) {
    return EntityRange();
}

EntityRange select(MeshBase const &m, MeshEntityType entityType, box_type const &b) {
    auto blk = m.clone();
    blk->intersection(b);
    blk->deploy();

    if (blk->empty()) {
        return EntityRange();
    } else {
        auto id_box = blk->index_box();
        return EntityRange(EntityIdCoder::make_range(std::get<0>(id_box), std::get<1>(id_box), entityType));
    }
};
}
}  // namespace simpla { namespace mesh_as

#endif  // SIMPLA_MODELSELECT_H