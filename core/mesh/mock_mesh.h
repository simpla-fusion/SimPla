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
#include "policy/rect_mesh.h"
#include "policy/constant_metric.h"
#include "policy/calculate_mock.h"

namespace simpla
{


template<typename ...> class _Field;

template<typename ...> class Mesh;

template<typename CS> using MockMesh= Mesh<
		policy::Metric<CS, tags::constant_space>,
		policy::RectMesh, policy::MockCalculate, policy::TimeIntegrator>;


}// namespace simpla
#endif //SIMPLA_MOCK_MESH_H
