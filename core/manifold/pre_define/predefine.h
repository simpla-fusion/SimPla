/**
 * @file predefine.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_PREDEFINE_H
#define SIMPLA_PREDEFINE_H

#include "../../gtl/primitives.h"

#include "../time_integrator/time_integrator.h"
#include "../topology/structured.h"
#include "../calculate/fvm_structured.h"
#include "../interpolate/linear.h"
#include "../geometry/geometry.h"

#include "../manifold.h"
#include "../domain.h"
#include "../domain_traits.h"

#include "../policy/dataset.h"
#include "../policy/parallel.h"

namespace simpla
{
template<typename CS, typename TOPOLOGY=topology::tags::CoRectMesh> using DefaultManifold= Manifold<
        Geometry<CS, TOPOLOGY>,
        Calculate<Geometry<CS, TOPOLOGY>, calculate::tags::finite_volume>,
        Interpolate<Geometry<CS, TOPOLOGY>, interpolate::tags::linear>,
        TimeIntegrator<Geometry<CS, TOPOLOGY>>,
        DataSetPolicy<Geometry<CS, TOPOLOGY>>,
        ParallelPolicy<Geometry<CS, TOPOLOGY>>
>;
}// namespace simpla
#endif //SIMPLA_PREDEFINE_H
