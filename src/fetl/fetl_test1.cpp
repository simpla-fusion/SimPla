/*
 * test_field.cpp
 *
 *  Created on: 2012-1-13
 *      Author: salmon
 */
#include "fetl_test1.h"
#include "fetl.h"
#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/geometry_euclidean.h"
#include "../mesh/traversal.h"

typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;

DEFINE_FIELDS(mesh_type)

typedef testing::Types<

Form<0>, Form<1>, Form<2>, Form<3>,

CForm<0>, CForm<1>, CForm<2>, CForm<3>,

VectorForm<0>, VectorForm<1>, VectorForm<2>, VectorForm<3>,

CVectorForm<0>, CVectorForm<1>, CVectorForm<2>, CVectorForm<3>,

TensorForm<0>, TensorForm<1>, TensorForm<2>, TensorForm<3>,

CTensorForm<0>, CTensorForm<1>, CTensorForm<2>, CTensorForm<3>

> AllFieldTypes;


INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETL, AllFieldTypes);
