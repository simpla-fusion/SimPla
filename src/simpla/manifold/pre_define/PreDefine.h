/**
 * @file predefine.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_PREDEFINE_H
#define SIMPLA_PREDEFINE_H

#include <simpla/geometry/CartesianGeometry.h>
#include <simpla/geometry/CylindricalGeometry.h>

//#include "../RectMesh.h"

#include "../Manifold.h"
#include "../schemes/FVMStructured.h"
#include "../schemes/LinearInterpolator.h"


namespace simpla { namespace manifold
{
//template<typename MESH, template<typename> class ...Policies>
//using ManifoldWithPolicies= CoordinateSystem<MESH, Policies<MESH>...>;

template<typename MESH = mesh::CartesianGeometry>
using DefaultManifold= Manifold<MESH,
        schemes::FiniteVolume,
        schemes::LinearInterpolator
        //        policy::StoragePolicy,
        //        policy::ParallelPolicy,
>;

using CylindricalManifold= DefaultManifold<mesh::CylindricalGeometry>;

using CartesianManifold = DefaultManifold<mesh::CartesianGeometry>;


}}// namespace simpla { namespace CoordinateSystem




#endif //SIMPLA_PREDEFINE_H
