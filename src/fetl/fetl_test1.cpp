/*
 * test_field.cpp
 *
 *  Created on: 2012-1-13
 *      Author: salmon
 */

#include "fetl_test.h"
#include "fetl_test1.h"

typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

typedef testing::Types<

TestFETLParam1<TMesh, Real, VERTEX>,

TestFETLParam1<TMesh, Real, EDGE>,

TestFETLParam1<TMesh, Real, FACE>,

TestFETLParam1<TMesh, Real, VOLUME>,

TestFETLParam1<TMesh, Complex, VERTEX>,

TestFETLParam1<TMesh, Complex, EDGE>,

TestFETLParam1<TMesh, Complex, FACE>,

TestFETLParam1<TMesh, Complex, VOLUME>,

TestFETLParam1<TMesh, nTuple<3, Real>, VERTEX>,

TestFETLParam1<TMesh, nTuple<3, Real>, EDGE>,

TestFETLParam1<TMesh, nTuple<3, Real>, FACE>,

TestFETLParam1<TMesh, nTuple<3, Real>, VOLUME>,

TestFETLParam1<TMesh, nTuple<3, Complex>, VERTEX>,

TestFETLParam1<TMesh, nTuple<3, Complex>, EDGE>,

TestFETLParam1<TMesh, nTuple<3, Complex>, FACE>,

TestFETLParam1<TMesh, nTuple<3, Complex>, VOLUME>,

TestFETLParam1<TMesh, nTuple<3, nTuple<3, Real>>, VERTEX>,

TestFETLParam1<TMesh, nTuple<3, nTuple<3, Real>>, EDGE>,

TestFETLParam1<TMesh, nTuple<3, nTuple<3, Real>>, FACE>,

TestFETLParam1<TMesh, nTuple<3, nTuple<3, Real>>, VOLUME>,

TestFETLParam1<TMesh, nTuple<3, nTuple<3, Complex>>, VERTEX>,

TestFETLParam1<TMesh, nTuple<3, nTuple<3, Complex>>, EDGE>,

TestFETLParam1<TMesh, nTuple<3, nTuple<3, Complex>>, FACE>,

TestFETLParam1<TMesh, nTuple<3, nTuple<3, Complex>>, VOLUME>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETL, ParamList);
