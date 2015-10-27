/**
 * @file riemannian.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_RIEMANNIAN_MESH_H
#define SIMPLA_RIEMANNIAN_MESH_H

#include "../../geometry/cs_cartesian.h"
#include "predefine.h"

namespace simpla
{
namespace manifold
{


template<int NDIMS> using Riemannian= DefaultManifold<coordinate_system::Cartesian<NDIMS, 2>>;

//Manifold<
//        CartesianCoordinate<NDIMS>,
//        Calculate<CartesianCoordinate<NDIMS>, calculate::tags::finite_volume>,
//        Interpolate<CartesianCoordinate<NDIMS>, interpolate::tags::linear>,
//        TimeIntegrator<CartesianCoordinate<NDIMS>>,
//        DataSetPolicy<CartesianCoordinate<NDIMS>>,
//        ParallelPolicy<CartesianCoordinate<NDIMS>>
//>;
}//namespace manifold
}//namespace simpla

#endif //SIMPLA_RIEMANNIAN_MESH_H
