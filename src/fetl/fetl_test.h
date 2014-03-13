/*
 * fetl_test.h
 *
 *  Created on: 2014年2月20日
 *      Author: salmon
 */

#ifndef FETL_TEST_H_
#define FETL_TEST_H_
#include <gtest/gtest.h>
#include <random>

#include "fetl.h"

#include "ntuple.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

#include "../mesh/rect_mesh.h"
#include "../mesh/octree_forest.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/geometry_euclidean.h"

#define DEF_MESH RectMesh<OcForest,CylindricalGeometry>

#endif /* FETL_TEST_H_ */
