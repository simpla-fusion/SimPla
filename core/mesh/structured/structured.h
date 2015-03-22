/*
 * @file cartesian_mesh.h
 *
 *  Created on: 2015年3月10日
 *      Author: salmon
 */

#ifndef CORE_MESH_STRUCTURED_STRUCTURED_H_
#define CORE_MESH_STRUCTURED_STRUCTURED_H_

#include "../mesh.h"
#include "../manifold.h"
#include "rect_mesh .h"
#include "coordiantes_cartesian.h"
//#include "coordinates_cylindrical.h"
#include "fdm.h"
#include "interpolator.h"

namespace simpla
{
typedef CartesianCoordinates<RectMesh> CartesianMesh;
//typedef CylindricalCoordinates<RectMesh> CylindricalMesh;

template<size_t IFORM> using CartesianManifold=
Manifold<IFORM,CartesianMesh,FiniteDiffMethod<CartesianMesh>,
InterpolatorLinear<CartesianMesh>>;

//template<size_t IFORM> using CylindricalMeshManifold=
//Manifold<IFORM,CartesianMesh,FiniteDiffMethod<CartesianMesh>,
//InterpolatorLinear<CartesianMesh>>;
}
// namespace simpla

#endif /* CORE_MESH_STRUCTURED_STRUCTURED_H_ */
