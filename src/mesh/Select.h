//
// Created by salmon on 16-6-2.
//

#ifndef SIMPLA_SELECT_H
#define SIMPLA_SELECT_H

#include "MeshBase.h"
#include "MeshEntity.h"

namespace simpla { namespace mesh
{

MeshEntityRange select(MeshBase const &m, MeshEntityRange const &, std::function<bool(point_type const &x)>);
}}//namespace simpla{namespace mesh{

#endif //SIMPLA_SELECT_H
