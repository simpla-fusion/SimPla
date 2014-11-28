/*
 * field_basic_algerbra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <iostream>
#include <gtest/gtest.h>

#include "../manifold/dummy_manifold.h"
#include "../manifold/domain.h"
#include "field.h"
#include "field_basic_algerbra_test.h"

using namespace simpla;

typedef testing::Types< //

		Field<Domain<DummyManifold>, double> //
		, Field<Domain<DummyManifold>, nTuple<double, 3> > //
		, Field<Domain<DummyManifold>, nTuple<double, 3, 3> > //
		, Field<Domain<DummyManifold>, nTuple<std::complex<double>, 3> > //

> TypeParamList;
template<typename TF> std::shared_ptr<typename TestField<TF>::manifold_type> //
TestField<TF>::manifold = std::make_shared<DummyManifold>(10, 20);

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

