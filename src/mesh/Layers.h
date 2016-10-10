//
// Created by salmon on 16-10-5.
//

#ifndef SIMPLA_MESHLAYERS_H
#define SIMPLA_MESHLAYERS_H

#include <type_traits>
#include "../toolbox/Log.h"
#include "../toolbox/nTuple.h"

#include "MeshCommon.h"
#include "Block.h"
#include "Atlas.h"

namespace simpla { namespace mesh
{
class Layers
{

    std::vector<Atlas> m_layers_;

};

}}//namespace simpla { namespace mesh

#endif //SIMPLA_MESHLAYERS_H
