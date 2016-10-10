//
// Created by salmon on 16-10-10.
//

#ifndef SIMPLA_MODELSELECT_H
#define SIMPLA_MODELSELECT_H

#include "../sp_def.h"
#include <functional>
#include "MeshCommon.h"
#include "Block.h"

namespace simpla { namespace mesh
{

template<typename TM, typename ...Args>
EntityRange select(TM const &m, MeshEntityType iform, Args &&...args);
}}//namespace simpla { namespace mesh

#endif //SIMPLA_MODELSELECT_H
