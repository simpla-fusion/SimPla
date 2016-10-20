/*
 * fetl_cylindrical_kz_test.cpp
 *
 *  created on: 2014-6-23
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include "../../diff_geometry/diff_scheme/fdm.h"
#include "../../diff_geometry/geometry/cylindrical.h"
#include "../../diff_geometry/interpolator/interpolator.h"
#include "../../diff_geometry/mesh.h"
#include "../../diff_geometry/topology/structured.h"

#include "../toolbox/ntuple.h"


using namespace simpla;

#define TMESH   Manifold<CylindricalCoordinate<StructuredMesh, CARTESIAN_ZAXIS>, FiniteDiffMehtod, InterpolatorLinear>

#include   "field_diff_calculus_test.h"

INSTANTIATE_TEST_CASE_P(FETLCylindrical, TestFETL,

testing::Combine(

testing::Values(

Vec3( { 10, -2.0, 0.0 })  //
//        , Vec3( { -1.0, -2.0, -3.0 })
		),

testing::Values(

Vec3( { 12.0, 2.0, PI })  //
		, Vec3( { 11.0, 2.0, 0.0 }) //
		, Vec3( { 11.0, 0.0, TWOPI }) //
		, Vec3( { 11.0, 2.0, TWOPI }) //

		),

testing::Values(

IVec3( { 101, 101, 1 }) //
		, IVec3( { 32, 36, 20 })   //
		),

testing::Values(Vec3( { TWOPI, 4 * PI, 2.0 }))

));
