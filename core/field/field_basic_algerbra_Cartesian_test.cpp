/*
 * field_basic_algerbra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <iostream>
#include <gtest/gtest.h>

#include "../manifold/manifold.h"
#include "../manifold/geometry/cartesian.h"
#include "../manifold/topology/structured.h"
#include "../manifold/diff_scheme/fdm.h"
#include "../manifold/interpolator/interpolator.h"
#include "field.h"
#include "field_basic_algerbra_test.h"

using namespace simpla;

typedef Manifold<CartesianCoordinates<StructuredMesh, CARTESIAN_ZAXIS>,
		FiniteDiffMethod, InterpolatorLinear> manifold_type;

typedef testing::Types< //

		Field<double, Domain<manifold_type, VERTEX>> //
//		, Field<double, Domain<manifold_type, EDGE>> //
//		, Field<double, Domain<manifold_type, FACE>> //
//		, Field<double, Domain<manifold_type, VOLUME>> //
//
//		, Field<nTuple<double, 3>, Domain<manifold_type, VERTEX>> //s
//		, Field<nTuple<double, 3>, Domain<manifold_type, EDGE>> //
//		, Field<nTuple<double, 3>, Domain<manifold_type, FACE>> //
//		, Field<nTuple<double, 3>, Domain<manifold_type, VOLUME>> //
//
//		, Field<nTuple<std::complex<double>, 3, 3>, Domain<manifold_type, VERTEX>> //
//		, Field<nTuple<std::complex<double>, 3, 3>, Domain<manifold_type, EDGE>> //
//		, Field<nTuple<double, 3, 3>, Domain<manifold_type, FACE>> //
//		, Field<nTuple<double, 3, 3>, Domain<manifold_type, VOLUME>> //

> TypeParamList;

//#define DECLARE_STATIC_MANIFOLD( _VALUE_TYPE_,_IFORM_ )                            \
//template<> std::shared_ptr<manifold_type>                                          \
//TestField<Field<_VALUE_TYPE_, Domain<manifold_type, _IFORM_> >>::manifold =        \
//		std::make_shared<manifold_type>(nTuple<Real, 3>( { 0.0, 0.0, 0.0 }),       \
//				nTuple<Real, 3>( { 1.0, 2.0, 1.0 }), nTuple<size_t, 3>( { 40,      \
//						12, 10 }));
//
//DECLARE_STATIC_MANIFOLD(double, VERTEX)
//DECLARE_STATIC_MANIFOLD(double, EDGE)
//DECLARE_STATIC_MANIFOLD(double, FACE)
//DECLARE_STATIC_MANIFOLD(double, VOLUME)

template<typename TF>
std::shared_ptr<typename TestField<TF>::manifold_type> TestField<TF>::manifold = //
		std::make_shared<manifold_type>( //
				nTuple<Real, 3>( { 0.0, 0.0, 0.0 }), //
				nTuple<Real, 3>( { 1.0, 2.0, 1.0 }), //
				nTuple<size_t, 3>( { 40, 12, 10 }) //
						);

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

