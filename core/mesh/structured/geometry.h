/**
 * @file geometry.h
 *
 * @date 2015年3月3日
 * @author salmon
 */

#ifndef CORE_MESH_STRUCTURED_GEOMETRY_H_
#define CORE_MESH_STRUCTURED_GEOMETRY_H_
#include "coordinates/cartesian.h"
//#include "coordinates/cylindrical.h"
#include "topology/structured.h"
namespace simpla
{

typedef CartesianCoordinates<StructuredMesh> CartesianMesh;
//typedef CylindricalCoordinates<StructuredMesh> CylindricalGeometry;

}// namespace simpla

#endif /* CORE_MESH_STRUCTURED_GEOMETRY_H_ */
