/**
 * @file riemannian.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_RIEMANNIAN_MESH_H
#define SIMPLA_RIEMANNIAN_MESH_H

#include "../../gtl/primitives.h"
#include "../../geometry/cs_cartesian.h"

#include "../time_integrator/time_integrator.h"
#include "../topology/structured.h"
#include "../calculate/fvm_structured.h"
#include "../interpolate/linear.h"
#include "../geometry/geometry.h"


#include "../manifold.h"

namespace simpla
{
namespace manifold
{
template<int NDIMS> using CartesianCoordinate= Geometry<geometry::coordinate_system::Cartesian<NDIMS, 2>, topology::tags::CoRectMesh>;
template<int NDIMS> using Riemannian= Manifold<
		CartesianCoordinate<NDIMS>,
		Calculate<CartesianCoordinate<NDIMS>, calculate::tags::finite_volume>,
		Interpolate<CartesianCoordinate<NDIMS>, interpolate::tags::linear>,
		TimeIntegrator<CartesianCoordinate<NDIMS>>
>;
}//namespace manifold
}//namespace simpla

#endif //SIMPLA_RIEMANNIAN_MESH_H
