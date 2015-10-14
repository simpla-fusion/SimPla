/**
 * @file mock_mesh.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_MOCK_MESH_H
#define SIMPLA_MOCK_MESH_H


#include "../gtl/primitives.h"
#include "policy/time_integrator.h"
#include "structured/mesh_aux.h"
#include "structured/rect_mesh.h"


#include "../geometry/cs_cartesian.h"
#include "topology/structured.h"
#include "calculate/fvm_structured.h"

#include "interpolate/linear.h"

#include "manifold/manifold.h"
#include "mesh.h"


namespace simpla
{


template<typename CS> using MockMesh= Mesh<

		Manifold<geometry::CartesianCoordinatesSystem<3>, topology::tags::CoRectMesh>,

		calculate::tags::finite_volume,

		interpolate::tags::linear,

		policy::TimeIntegrator


>;

}// namespace simpla
#endif //SIMPLA_MOCK_MESH_H
