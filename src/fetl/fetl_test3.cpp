/*
 * fetl_test3.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include "fetl_test3.h"

#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/geometry_euclidean.h"

typedef testing::Types<RectMesh<OcForest, EuclideanGeometry> > MeshTypes;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETLDiffCalcuate1, MeshTypes);

typedef testing::Types<Field<RectMesh<OcForest, EuclideanGeometry>, VERTEX, Real>,
		Field<RectMesh<OcForest, EuclideanGeometry>, VERTEX, Complex>,
		Field<RectMesh<OcForest, EuclideanGeometry>, VERTEX, nTuple<3, Real>> > FieldTypes;
INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETLDiffCalcuate, FieldTypes);
