/**
 * @file predefine.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_PREDEFINE_H
#define SIMPLA_PREDEFINE_H

#include "../../gtl/primitives.h"

#include "../time_integrator/time_integrator.h"
#include "../diff_scheme/fvm_structured.h"
#include "../interpolate/linear.h"
#include "../base_manifold.h"

#include "../manifold.h"
#include "../domain.h"
#include "../domain_traits.h"

#include "../policy/dataset.h"
#include "../policy/parallel.h"

namespace simpla
{
namespace manifold
{
template<typename METRIC, typename MESH> using DefaultManifold= Manifold<
        BaseManifold<METRIC, MESH>,
        DiffScheme<BaseManifold<METRIC, MESH>, diff_scheme::tags::finite_volume>,
        Interpolate<BaseManifold<METRIC, MESH>, interpolate::tags::linear>,
        TimeIntegrator<BaseManifold<METRIC, MESH>>,
        DataSetPolicy<BaseManifold<METRIC, MESH>>,
        ParallelPolicy<BaseManifold<METRIC, MESH>>
>;
}; //namespace manifold
}// namespace simpla
#endif //SIMPLA_PREDEFINE_H
