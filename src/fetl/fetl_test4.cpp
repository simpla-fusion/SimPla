/*
 * fetl_test4.cpp
 *
 *  Created on: 2014年3月11日
 *      Author: salmon
 */

#include "fetl_test4.h"

#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/geometry_euclidean.h"

typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;
DEFINE_FIELDS(mesh_type)

typedef testing::Types<Field<mesh_type, VERTEX, Real>, Field<mesh_type, VERTEX, Complex> > AllFieldTypes;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETLVecField, AllFieldTypes);

