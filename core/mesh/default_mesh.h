/**
 * @file default_mesh.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_DEFAULT_MESH_H
#define SIMPLA_DEFAULT_MESH_H

#include "../gtl/primitives.h"
#include "mesh.h"
#include "policy/time_integrator.h"
#include "structured/mesh_aux.h"
#include "structured/rect_mesh.h"
#include "geometry/geometry.h"

#include "topology/structured.h"

#include "fvm_structured.h"
#include "structured/interpolate.h"

namespace simpla
{


template<typename CS> using DefaultMesh= Mesh<
		geometry::Geometry<CS, topology::CoRectMesh> >,
calculate::tags::finite_volume, interpolator::tags::linear, policy::TimeIntegrator,
policy::MeshUtilities<policy::Geometry<CS, topology::CoRectMesh> >
>;

}
#endif //SIMPLA_DEFAULT_MESH_H
