/*
 * fetl_test2.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include "fetl_test.h"
#include "fetl_test2.h"

#include "../mesh/mesh.h"

typedef Mesh<EuclideanGeometry<OcForest<>>> TMesh;

typedef testing::Types<

TestFETLParam2<Mesh<EuclideanGeometry<OcForest<Real>>> , Real, VERTEX>,

TestFETLParam2<Mesh<EuclideanGeometry<OcForest<Real>>>, Complex, VERTEX>,

TestFETLParam2<Mesh<EuclideanGeometry<OcForest<std::complex<Real> >>>, Real, VERTEX>,

TestFETLParam2<Mesh<EuclideanGeometry<OcForest<std::complex<Real>>>>, Complex, VERTEX>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETLVecAlgegbra, ParamList);
