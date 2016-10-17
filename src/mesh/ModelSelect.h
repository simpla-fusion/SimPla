//
// Created by salmon on 16-10-10.
//

#ifndef SIMPLA_MODELSELECT_H
#define SIMPLA_MODELSELECT_H

#include "../sp_def.h"
#include <functional>
#include "MeshCommon.h"
#include "MeshBase.h"

namespace simpla { namespace mesh
{

template<typename TM, typename ...Args>
EntityRange select(TM const &m, MeshEntityType iform, Args &&...args) { return EntityRange(); }


EntityRange select(MeshBase const &m, box_type const &b, MeshEntityType entityType = VERTEX)
{

    auto blk = m.clone();
    blk->intersection(b);
    blk->deploy();

    if (blk->empty())
    {
        return EntityRange();
    } else
    {
        auto id_box = blk->index_box();
        return EntityRange(
                MeshEntityIdCoder::make_range(std::get<0>(id_box), std::get<1>(id_box), entityType));
    }

};
}}//namespace simpla { namespace mesh

#endif //SIMPLA_MODELSELECT_H
