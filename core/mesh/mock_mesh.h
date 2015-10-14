/**
 * @file mock_mesh.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_MOCK_MESH_H
#define SIMPLA_MOCK_MESH_H

#include <stddef.h>
#include <memory>
#include "../gtl/primitives.h"
#include "mesh.h"
#include "policy/time_integrator.h"
#include "policy/mock_policies.h"
#include "structured/mesh_aux.h"
#include "structured/rect_mesh.h"
#include "geometry.h"


namespace simpla
{


template<typename ...> class Field;

template<typename ...> class Mesh;

template<typename CS> using MockMesh= Mesh<
		policy::Geometry<CS, policy::RectMesh, tags::constant_space>,
		tags::is_mock, tags::is_mock, policy::TimeIntegrator,
		policy::MeshUtilities<policy::Geometry<CS, policy::RectMesh, tags::constant_space> >
>;


}// namespace simpla
#endif //SIMPLA_MOCK_MESH_H
