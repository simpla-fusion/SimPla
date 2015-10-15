/**
 * @file cylindrical.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_MANIFOLD_CYLINDRICAL_H
#define SIMPLA_MANIFOLD_CYLINDRICAL_H

#include "../../gtl/primitives.h"
#include "../../geometry/cs_cylindrical.h"

#include "../time_integrator/time_integrator.h"
#include "../topology/structured.h"
#include "../calculate/fvm_structured.h"
#include "../interpolate/linear.h"
#include "../geometry/geometry.h"

#include "../policy/dataset.h"
#include "../manifold.h"

namespace simpla
{
namespace manifold
{
using CylindricalCoordinate= Geometry<geometry::coordinate_system::Cylindrical<2>, topology::tags::CoRectMesh>;
using CylindricalCoRect= Manifold<
		CylindricalCoordinate,
		Calculate<CylindricalCoordinate, calculate::tags::finite_volume>,
		Interpolate<CylindricalCoordinate, interpolate::tags::linear>,
		TimeIntegrator<CylindricalCoordinate>,
		DataSetPolicy<CylindricalCoordinate>
>;
}//namespace  manifold
}//namespace simpla
#endif //SIMPLA_MANIFOLD_CYLINDRICAL_H
