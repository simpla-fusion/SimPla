//
// Created by salmon on 16-6-28.
//

#ifndef SIMPLA_CONTEXTUTILITY_H
#define SIMPLA_CONTEXTUTILITY_H

#include "Context.h"

namespace simpla { namespace simulation
{

template<typename TProb>
void extent_span(Context *ctx, mesh::MeshBlockId center, size_type width)
{
    auto &atlas = ctx->atlas();

    int od[3];
    for (int tag = 1, tag_e = 1 << 6; tag < tag_e; tag <<= 1)
    {

        od[0] = ((tag & 0x3) << 1) - 3;
        od[1] = (((tag >> 2) & 0x3) << 1) - 3;
        od[2] = (((tag >> 4) & 0x3) << 1) - 3;

        if (od[0] > 1 || od[1] > 1 || od[2] > 1)
        {
            continue;
        }
        auto id = atlas.extent_block(center, od, width);

        ctx->add_domain_as<TProb>(id, od);
    }


}


}}//namespace simpla{namespace simulation{

#endif //SIMPLA_CONTEXTUTILITY_H
