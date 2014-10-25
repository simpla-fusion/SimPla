/*
 * interpolator_test.cpp
 *
 *  created on: 2014-6-29
 *      Author: salmon
 */

#include "interpolator_test.h"

#include "../geometry/cartesian.h"
#include "../topology/structured.h"
#include "../domain.h"

using namespace simpla;

typedef testing::Types<InterpolatorLinear<CartesianCoordinates<StructuredMesh>> > DomainList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestInterpolator, DomainList);
