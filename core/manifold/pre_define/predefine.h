/**
 * @file predefine.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_PREDEFINE_H
#define SIMPLA_PREDEFINE_H

#include "../../gtl/primitives.h"

#include "../time_integrator/time_integrator.h"
#include "../calculate/fvm_structured.h"
#include "../interpolate/linear.h"
#include "../base_manifold.h"

#include "../manifold.h"
#include "../domain.h"
#include "../domain_traits.h"

#include "../policy/dataset.h"
#include "../policy/parallel.h"

namespace simpla
{
template<typename METRIC, typename TOPOLOGY> using DefaultManifold= Manifold<
        BaseManifold<METRIC, TOPOLOGY>,
        Calculate<BaseManifold<METRIC, TOPOLOGY>, calculate::tags::finite_volume>,
        Interpolate<BaseManifold<METRIC, TOPOLOGY>, interpolate::tags::linear>,
        TimeIntegrator<BaseManifold<METRIC, TOPOLOGY>>,
        DataSetPolicy<BaseManifold<METRIC, TOPOLOGY>>,
        ParallelPolicy<BaseManifold<METRIC, TOPOLOGY>>
>;
}// namespace simpla
#endif //SIMPLA_PREDEFINE_H
