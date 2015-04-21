/*
 * @file cartesian_mesh.h
 *
 *  Created on: 2015年3月10日
 *      Author: salmon
 */

#ifndef CORE_MESH_STRUCTURED_STRUCTURED_H_
#define CORE_MESH_STRUCTURED_STRUCTURED_H_

#include "../mesh.h"
#include "../mesh_ids.h"
#include "rect_mesh.h"
#include "coordiantes_cartesian.h"
//#include "coordinates_cylindrical.h"
#include "fdm.h"
#include "interpolator.h"

namespace simpla
{

typedef RectMesh<MeshIDs_<3>, CartesianCoordinates<3>, InterpolatorLinear,
		FiniteDiffMethod> CartesianRectMesh;

}
// namespace simpla

#endif /* CORE_MESH_STRUCTURED_STRUCTURED_H_ */
