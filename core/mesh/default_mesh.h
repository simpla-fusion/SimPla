/**
 * @file default_mesh.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_DEFAULT_MESH_H
#define SIMPLA_DEFAULT_MESH_H

#include "structured/rect_mesh.h"
#include "structured/interpolate.h"
#include "structured/calculate.h"

namespace simpla
{
template<typename CS> using DefaultMesh= Mesh<CS, tags::RectMesh, tags::finite_difference, tags::linear>;

}
#endif //SIMPLA_DEFAULT_MESH_H
