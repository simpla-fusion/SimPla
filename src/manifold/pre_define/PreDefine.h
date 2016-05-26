/**
 * @file predefine.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_PREDEFINE_H
#define SIMPLA_PREDEFINE_H

#include "../../mesh/CoRectMesh.h"

#include "../Manifold.h"
#include "../schemes/FVMStructured.h"
#include "../schemes/LinearInterpolator.h"

//#include "../policy/StoragePolicy.h"
//#include "../policy/ParallelPolicy.h"

#include "../metric/Cartesian.h"
#include "../metric/Cylindrical.h"

#include "../../mesh/CoRectMesh.h"

namespace simpla { namespace manifold
{
//template<typename MESH, template<typename> class ...Policies>
//using ManifoldWithPolicies= Manifold<MESH, Policies<MESH>...>;

template<typename MESH = mesh::CoRectMesh, template<typename> class METRIC_POLICY= metric::Cartesian>
using DefaultManifold= Manifold<MESH,
        METRIC_POLICY,
        schemes::FiniteVolume,
        schemes::LinearInterpolator
        //        policy::StoragePolicy,
        //        policy::ParallelPolicy,
>;

using CylindricalManifold= DefaultManifold<mesh::CoRectMesh, metric::Cylindrical>;

using CartesianManifold = DefaultManifold<mesh::CoRectMesh, metric::Cartesian>;


}}// namespace simpla { namespace Manifold

namespace simpla { namespace tags { struct function; }}

namespace simpla { namespace traits
{
template<typename ValueType, typename TM, int IFORM = VERTEX>
using field_t=  Field<ValueType, TM, std::integral_constant<int, IFORM> >;

//template<typename TV, int I, typename TM> field_t<TV, TM, I>
//make_field(TM const &mesh) { return field_t<TV, TM, I>(mesh); };
//
//
//template<typename TV, typename TM, int IFORM, typename TFun>
//using field_function_t=Field<TV, TM, std::integral_constant<int, IFORM>,
//        tags::function, TFun, typename TM::box_type,
//        manifold::policy::FiniteVolume,
//        manifold::policy::LinearInterpolator>;
//
//template<typename TV, int IFORM, typename TM, typename TDict>
//field_function_t<TV, TM, IFORM, TDict> make_field_function(TM const &m, TDict const &dict)
//{
//    return field_function_t<TV, TM, IFORM, TDict>::create(m, dict);
//}
//
//template<typename TV, int IFORM, typename TM, typename TDict>
//field_function_t<TV, TM, IFORM, TDict> make_field_function_from_config(TM const &m, TDict const &dict)
//{
//    return field_function_t<TV, TM, IFORM, TDict>::create_from_config(m, dict);
//}
}}

#endif //SIMPLA_PREDEFINE_H
