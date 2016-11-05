/**
 * @file predefine.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_PREDEFINE_H
#define SIMPLA_PREDEFINE_H

#include "../../manifold/CoRectMesh.h"
#include "../../manifold/RectMesh.h"

#include "../Manifold.h"
#include "../schemes/FVMStructured.h"
#include "../schemes/LinearInterpolator.h"

//#include "../policy/StoragePolicy.h"
//#include "../policy/ParallelPolicy.h"

//#include "../metric/Cartesian.h"
//#include "../metric/Cylindrical.h"


namespace simpla { namespace manifold
{
//template<typename MESH, template<typename> class ...Policies>
//using ManifoldWithPolicies= CoordinateSystem<MESH, Policies<MESH>...>;

template<typename MESH = mesh::CoRectMesh>
using DefaultManifold= Manifold<MESH,
        schemes::FiniteVolume,
        schemes::LinearInterpolator
        //        policy::StoragePolicy,
        //        policy::ParallelPolicy,
>;

using CylindricalManifold= DefaultManifold<mesh::RectMesh>;

using CartesianManifold = DefaultManifold<mesh::CoRectMesh>;


}}// namespace simpla { namespace CoordinateSystem

namespace simpla { namespace tags { struct function; }}

namespace simpla { namespace traits
{
template<typename ValueType, typename TM, int IFORM = mesh::VERTEX>
using field_t=  Field<ValueType, TM, std::integral_constant<int, IFORM> >;

//template<typename TV, int I, typename TM> field_t<TV, TM, I>
//make_field(TM const &mesh_as) { return field_t<TV, TM, I>(get_mesh); };
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
