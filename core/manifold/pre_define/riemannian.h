/**
 * @file riemannian.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_DEFAULT_MESH_H
#define SIMPLA_DEFAULT_MESH_H

#include "../../gtl/primitives.h"
#include "../../geometry/cs_cartesian.h"

#include "../policy/time_integrator.h"
#include "../topology/structured.h"
#include "../calculate/fvm_structured.h"
#include "../interpolate/linear.h"
#include "../geometry/geometry.h"


#include "../manifold.h"


namespace simpla
{
namespace manifold
{
template<int NDIMS> using Riemannian= Manifold<

		Geometry<geometry::coordinate_system::Cartesian<NDIMS, 2>, topology::tags::CoRectMesh>,

		calculate::tags::finite_volume,

		interpolate::tags::linear,

		policy::TimeIntegrator


>;
}//namespace  manifold
}//namespace simpla

#endif //SIMPLA_DEFAULT_MESH_H
