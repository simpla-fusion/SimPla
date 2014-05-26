/*
 * fetl_test4.cpp
 *
 *  Created on: 2014年3月11日
 *      Author: salmon
 */
#include "fetl_test4.h"

typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

typedef testing::Types<

TestFETLParam4<TMesh, Real, VERTEX>,

TestFETLParam4<TMesh, Complex, VERTEX>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETLVecField, ParamList);

