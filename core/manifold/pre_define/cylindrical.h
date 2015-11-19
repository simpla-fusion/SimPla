/**
 * @file cylindrical.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_MANIFOLD_CYLINDRICAL_H
#define SIMPLA_MANIFOLD_CYLINDRICAL_H

#include "predefine.h"
#include "../../geometry/coordinate_system.h"
#include "../../geometry/cs_cylindrical.h"
#include "../mesh/rectmesh.h"

namespace simpla
{
namespace manifold
{
using Cylindrical= DefaultManifold<geometry::Metric<geometry::coordinate_system::Cylindrical<2> >, mesh::RectMesh<>>;

}//namespace  manifold
}//namespace simpla
#endif //SIMPLA_MANIFOLD_CYLINDRICAL_H
