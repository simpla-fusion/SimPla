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

#include "../policy/time_integrator_policy.h"
#include "../policy/fvm_structured_policy.h"
#include "../policy/linear_interpolator_policy.h"
#include "../policy/storage_policy.h"
#include "../policy/parallel_policy.h"

namespace simpla { namespace manifold
{
template<typename MESH>
using DefaultManifold= Manifold<
        MESH,
        policy::FiniteVolume<MESH>,
        policy::LinearInterpolator<MESH>,
        policy::StoragePolicy<MESH>,
        policy::ParallelPolicy<MESH>
>;


using CylindricalManifold= DefaultManifold<mesh::RectMesh<geometry::CylindricalMetric >>;

using CartesianManifold=DefaultManifold<mesh::CoRectMesh<geometry::CartesianMetric> >;

}}// namespace simpla { namespace manifold

#endif //SIMPLA_PREDEFINE_H
