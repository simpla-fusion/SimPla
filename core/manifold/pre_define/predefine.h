/**
 * @file predefine.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_PREDEFINE_H
#define SIMPLA_PREDEFINE_H


#include "../manifold.h"

#include "../../geometry/coordinate_system.h"
#include "../../geometry/cs_cartesian.h"
#include "../../geometry/cs_cylindrical.h"
#include "../mesh/rect_mesh.h"
#include "../mesh/corect_mesh.h"

#include "../time_integrator/time_integrator.h"
#include "../diff_scheme/fvm_structured.h"
#include "../interpolate/linear.h"
#include "../policy/storage.h"
#include "../policy/parallel.h"


namespace simpla { namespace manifold
{
template<typename MESH>
using DefaultManifold= Manifold<
        MESH,
        policy::DiffScheme<MESH, policy::diff_scheme::tags::finite_volume>,
        policy::Interpolate<MESH, policy::interpolate::tags::linear>,
        policy::TimeIntegrator<MESH>,
        policy::StoragePolicy<MESH>,
        policy::ParallelPolicy<MESH>
>;


using CylindricalManifold= DefaultManifold<mesh::RectMesh<geometry::CylindricalMetric >>;

using CartesianManifold=DefaultManifold<mesh::CoRectMesh<geometry::CartesianMetric> >;

}}// namespace simpla { namespace manifold

#endif //SIMPLA_PREDEFINE_H
