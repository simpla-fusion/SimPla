/**
 * @file mock_mesh.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_MOCK_MESH_H
#define SIMPLA_MOCK_MESH_H


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
template<int NDIMS> using Mock= Manifold<

		Geometry<geometry::coordinate_system::Cartesian<NDIMS, 2>, topology::tags::CoRectMesh>,

		calculate::tags::finite_volume,

		interpolate::tags::linear,

		policy::TimeIntegrator


>;

template<int NDIMS> using CartesianCoordinate= Geometry<geometry::coordinate_system::Cartesian<NDIMS, 2>, topology::tags::CoRectMesh>;
template<int NDIMS> using Riemannian= Manifold<
		CartesianCoordinate<NDIMS>,
		calculate::Calculate<CartesianCoordinate<NDIMS>, calculate::tags::finite_volume>,
		interpolate::Interpolate<CartesianCoordinate<NDIMS>, interpolate::tags::linear>
>;
}//namespace  manifold
}//namespace simpla
#endif //SIMPLA_MOCK_MESH_H
