/**
 * @file predefine.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_PREDEFINE_H
#define SIMPLA_PREDEFINE_H


#include "../Manifold.h"

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
#include "../../field/Field.h"

namespace simpla { namespace manifold
{
//template<typename MESH, template<typename> class ...Policies>
//using ManifoldWithPolicies= Manifold<MESH, Policies<MESH>...>;

template<typename MESH>
using DefaultManifold= Manifold<MESH,
        policy::StoragePolicy,
        policy::ParallelPolicy>;

using CylindricalManifold= DefaultManifold<mesh::RectMesh<geometry::CylindricalMetric> >;

using CartesianManifold=DefaultManifold<mesh::CoRectMesh<geometry::CartesianMetric> >;


}}// namespace simpla { namespace Manifold

namespace simpla { namespace tags { struct function; }}

namespace simpla { namespace traits
{
template<typename ValueType, typename TM, int IFORM = VERTEX>
using field_t=  Field<ValueType, TM, std::integral_constant<int, IFORM>,
        manifold::policy::FiniteVolume < TM>,
manifold::policy::LinearInterpolator <TM>
>;

template<typename TV, int I, typename TM> field_t<TV, TM, I>
make_field(TM const &mesh) { return field_t<TV, TM, I>(mesh); };


template<typename TV, typename TM, int IFORM, typename TFun>
using field_function_t=Field<TV, TM, std::integral_constant<int, IFORM>,
        tags::function, TFun, typename TM::box_type,
        manifold::policy::FiniteVolume < TM>,
manifold::policy::LinearInterpolator <TM>>;

template<typename TV, int IFORM, typename TM, typename TDict>
field_function_t<TV, TM, IFORM, TDict> make_field_function(TM const &m, TDict const &dict)
{
    return field_function_t<TV, TM, IFORM, TDict>::create(m, dict);
}

template<typename TV, int IFORM, typename TM, typename TDict>
field_function_t<TV, TM, IFORM, TDict> make_field_function_from_config(TM const &m, TDict const &dict)
{
    return field_function_t<TV, TM, IFORM, TDict>::create_from_config(m, dict);
}
}}

#endif //SIMPLA_PREDEFINE_H
