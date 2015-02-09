/*
 * geometry_cartesian_test.cpp
 *
 *  created on: 2014-6-27
 *      Author: salmon
 */

#include "../../structured/structured.h"
#include "../coordinates/cartesian.h"

#define GEOMETRY  CartesianCoordinates<StructuredMesh>

#include "../../structured/test/geometry_test.h"

using namespace simpla;

INSTANTIATE_TEST_CASE_P(Test_CartesianGeometry, TestGeometry,

testing::Combine(

testing::Values(

nTuple<Real, 3>( { 1.0, 0.0, 0.0, }), //
nTuple<Real, 3>( { 1.0, -2.0, -3.0 }), //
nTuple<Real, 3>( { 1.0, 1.0, 1.0 }) //

		),

testing::Values(

nTuple<Real, 3>( { 2.0, 0.0, 0.0 }), //
nTuple<Real, 3>( { 0.0, 2.0, 0.0 }), //
nTuple<Real, 3>( { 0.0, 0.0, 2.0 }), //
nTuple<Real, 3>( { 0.0, 2.0, 2.0 }), //
nTuple<Real, 3>( { 2.0, 0.0, 2.0 }), //
nTuple<Real, 3>( { 2.0, 2.0, 0.0 }), //
nTuple<Real, 3>( { 1.0, 3.0, 2.0 })  //

		),

testing::Values(

nTuple<size_t, 3>( { 10, 1, 1 }),  //
nTuple<size_t, 3>( { 11, 1, 1 }), //
nTuple<size_t, 3>( { 1, 17, 1 }), //
nTuple<size_t, 3>( { 1, 1, 10 }), //
nTuple<size_t, 3>( { 1, 17, 17 }), //
nTuple<size_t, 3>( { 17, 1, 17 }), //
nTuple<size_t, 3>( { 17, 17, 1 }), nTuple<size_t, 3>( { 13, 16, 10 })   //

		)));
