/*
 * fetl_test2.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include "fetl_test2.h"

#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/geometry_euclidean.h"
#include "../mesh/traversal.h"

typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;

typedef testing::Types<Field<mesh_type, VERTEX, Real>, Field<mesh_type, VERTEX, Complex> > AllFieldTypes;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETLVecAlgegbra, AllFieldTypes);
