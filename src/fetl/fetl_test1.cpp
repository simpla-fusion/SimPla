/*
 * test_field.cpp
 *
 *  Created on: 2012-1-13
 *      Author: salmon
 */

#include "fetl_test.h"
#include "fetl_test1.h"

using ParamList=

testing::Types<

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, Real, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, Real, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, Real, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, Real, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, Complex, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, Complex, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, Complex, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, Complex, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, Real>, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, Real>, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, Real>, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, Real>, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, Complex>, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, Complex>, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, Complex>, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, Complex>, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, nTuple<3, Real>>, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, nTuple<3, Real>>, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, nTuple<3, Real>>, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, nTuple<3, Real>>, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, nTuple<3, Complex>>, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, nTuple<3, Complex>>, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, nTuple<3, Complex>>, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Real>>>, nTuple<3, nTuple<3, Complex>>, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, Real, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, Real, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, Real, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, Real, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, Complex, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, Complex, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, Complex, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, Complex, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, Real>, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, Real>, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, Real>, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, Real>, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, Complex>, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, Complex>, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, Complex>, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, Complex>, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, nTuple<3, Real>>, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, nTuple<3, Real>>, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, nTuple<3, Real>>, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, nTuple<3, Real>>, VOLUME>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, nTuple<3, Complex>>, VERTEX>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, nTuple<3, Complex>>, EDGE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, nTuple<3, Complex>>, FACE>,

TestFETLParam1<Mesh<EuclideanGeometry<OcForest<Complex>>>, nTuple<3, nTuple<3, Complex>>, VOLUME>

>;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETL, ParamList);
