/*
 * @file cartesian_mesh.h
 *
 *  Created on: 2015年3月10日
 *      Author: salmon
 */

#ifndef CORE_MESH_STRUCTURED_CARTESIAN_MESH_H_
#define CORE_MESH_STRUCTURED_CARTESIAN_MESH_H_

#include "../mesh.h"
#include "../manifold.h"
#include "topology/structured.h"
#include "coordinates/cartesian.h"
#include "diff_scheme/fdm.h"
#include "interpolator/interpolator.h"
namespace simpla
{
typedef CartesianCoordinates<StructuredMesh> CartesianMesh;

}  // namespace simpla

#endif /* CORE_MESH_STRUCTURED_CARTESIAN_MESH_H_ */
