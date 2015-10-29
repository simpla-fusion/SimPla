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
#include "../topology/topology.h"
#include "../topology/corect_mesh.h"

namespace simpla
{
namespace manifold
{
using Cylindrical= DefaultManifold<Metric<coordinate_system::Cylindrical<2> >, topology::CoRectMesh>;

}//namespace  manifold
}//namespace simpla
#endif //SIMPLA_MANIFOLD_CYLINDRICAL_H
