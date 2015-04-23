/*
 * field_basic_algerbra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../diff_geometry/fetl.h"

#include "../field.h"
#include "../field_basic_algerbra_test.h"

using namespace simpla;

typedef Manifold<CylindricalCoordinates<RectMesh, CARTESIAN_ZAXIS>,
		FiniteDiffMethod, InterpolatorLinear> manifold_type;

typedef testing::Types<
		//

		Field<Domain<manifold_type, VERTEX>, double> //
		,
		Field<Domain<manifold_type, EDGE>, double> //
		,
		Field<Domain<manifold_type, FACE>, double> //
		,
		Field<Domain<manifold_type, VOLUME>, double> //

		,
		Field<Domain<manifold_type, VERTEX>, nTuple<double, 3>> //
		,
		Field<Domain<manifold_type, EDGE>, nTuple<double, 3>> //
		,
		Field<Domain<manifold_type, FACE>, nTuple<double, 3>> //
		,
		Field<Domain<manifold_type, VOLUME>, nTuple<double, 3>> //

		,
		Field<Domain<manifold_type, VERTEX>, nTuple<std::complex<double>, 3, 3>> //
		, Field<Domain<manifold_type, EDGE>, nTuple<std::complex<double>, 3, 3>> //
		, Field<Domain<manifold_type, FACE>, nTuple<double, 3, 3>> //
		, Field<Domain<manifold_type, VOLUME>, nTuple<double, 3, 3>> //

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
std::shared_ptr<typename TestField<TF>::manifold_type> TestField<TF>::mesh = //
		std::make_shared<mesh_type>( //
				nTuple<Real, 3>( { 0.0, 0.0, 0.0 }), //
				nTuple<Real, 3>( { 1.0, 2.0, 1.0 }), //
				nTuple<size_t, 3>( { 40, 12, 10 }) //
						);

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

