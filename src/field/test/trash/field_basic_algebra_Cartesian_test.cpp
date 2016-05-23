/*
 * field_basic_algerbra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../diff_geometry/diff_scheme/fdm.h"
#include "../../diff_geometry/geometry/cartesian.h"
#include "../../diff_geometry/interpolator/interpolator.h"
#include "../../diff_geometry/mesh.h"
#include "../../diff_geometry/topology/structured.h"

#include "../../utilities/utilities.h"
#include "../field.h"
#include "field_basic_algebra_test.h"

using namespace simpla;

typedef Manifold<CartesianCoordinate<RectMesh, CARTESIAN_ZAXIS>,
		FiniteDiffMethod, InterpolatorLinear> manifold_type;

typedef testing::Types<
		//

		Field<double, manifold_type::template Domain<VERTEX>> //
		,
		Field<double, manifold_type::template Domain<EDGE>> //
		,
		Field<double, manifold_type::template Domain<FACE>> //
		,
		Field<double, manifold_type::template Domain<VOLUME>> //

		,
		Field<nTuple<double, 3>, manifold_type::template Domain<VERTEX>> //
		,
		Field<nTuple<double, 3>, manifold_type::template Domain<EDGE>> //
		,
		Field<nTuple<double, 3>, manifold_type::template Domain<FACE>> //
		,
		Field<nTuple<double, 3>, manifold_type::template Domain<VOLUME>> //

		,
		Field<nTuple<std::complex<double>, 3, 3>,
				manifold_type::template Domain<VERTEX>> //
		,
		Field<nTuple<std::complex<double>, 3, 3>,
				manifold_type::template Domain<EDGE>> //
		, Field<nTuple<double, 3, 3>, manifold_type::template Domain<FACE>> //
		, Field<nTuple<double, 3, 3>, manifold_type::template Domain<VOLUME>> //

> TypeParamList;

//#define DECLARE_STATIC_MANIFOLD( _VALUE_TYPE_,_IFORM_ )                            \
//template<> std::shared_ptr<manifold_type>                                          \
//TestField<field<_VALUE_TYPE_, manifold_type::template Domain< _IFORM_> >>::geometry =        \
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
				nTuple<double, 3>(
				{ 0.0, 0.0, 0.0 }), //
				nTuple<double, 3>(
				{ 1.0, 2.0, 1.0 }), //
				nTuple<size_t, 3>(
				{ 40, 12, 10 }) //
						);

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

