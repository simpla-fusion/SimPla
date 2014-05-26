/*
 * fetl_test2.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include "fetl_test.h"
#include "fetl_test2.h"

#include "../mesh/mesh.h"

typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

typedef testing::Types<

TestFETLParam2<TMesh, Real, VERTEX>,

TestFETLParam2<TMesh, Complex, VERTEX>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETLVecAlgegbra, ParamList);
