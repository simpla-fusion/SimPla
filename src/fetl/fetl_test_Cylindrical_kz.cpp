/*
 * fetl_test_Cylindrical_kz.cpp
 *
 *  Created on: 2014年6月23日
 *      Author: salmon
 */

#include "../mesh/mesh_rectangle.h"
#include "../mesh/octree_forest.h"
#include "../mesh/geometry_cylindrical.h"

#define TMESH Mesh<CylindricalGeometry<OcForest<std::complex<Real>>>>

#include "fetl_test.h"

