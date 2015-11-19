/**
 * @file mock.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_MOCK_H
#define SIMPLA_MOCK_H

#include "../../gtl/primitives.h"
#include "../../geometry/cs_cartesian.h"
#include "base_manifold.h"
#include "co_rect_mesh.h"
#include "../calculate/fvm_structured.h"
#include "../interpolate/linear.h"
#include "../manifold.h"
#include "storage.h"

namespace simpla
{
namespace manifold
{
using CartesianCoordinate= Geometry<geometry::coordinate_system::Cartesian<3, 2>, topology::tags::CoRectMesh>;

using Mock= Manifold<CartesianCoordinate, Calculate<CartesianCoordinate, calculate::tags::finite_volume>,
		Interpolate<CartesianCoordinate, interpolate::tags::linear>,
		DataSetPolicy<CartesianCoordinate>>;


}//namespace  manifold
}//namespace simpla
#endif //SIMPLA_MOCK_H
