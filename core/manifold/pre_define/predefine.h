/**
 * @file predefine.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_PREDEFINE_H
#define SIMPLA_PREDEFINE_H


#include "../time_integrator/time_integrator.h"
#include "../diff_scheme/fvm_structured.h"
#include "../interpolate/linear.h"

#include "../manifold.h"

#include "../policy/storage.h"
#include "../policy/parallel.h"

namespace simpla
{
namespace manifold
{
template<typename METRIC, typename MESH> using DefaultManifold= Manifold<
        MESH, METRIC,
        DiffScheme<MESH, diff_scheme::tags::finite_volume>,
        Interpolate<MESH, interpolate::tags::linear>,
        TimeIntegrator<MESH>,
        StoragePolicy<MESH>,
        ParallelPolicy<MESH>
>;
}; //namespace manifold
}// namespace simpla
#endif //SIMPLA_PREDEFINE_H
