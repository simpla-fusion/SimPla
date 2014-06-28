/*
 * fetl_cylindrical_kz_test.cpp
 *
 *  Created on: 2014年6月23日
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include "../mesh/mesh_rectangle.h"
#include "../mesh/octree_forest.h"
#include "../mesh/geometry_cylindrical.h"
#include "fetl.h"

#define TMESH Mesh<CylindricalGeometry<OcForest>>>

#include "fetl_test.h"
#include "fetl_test1.h"
#include "fetl_test2.h"
#include "fetl_test3.h"
#include "fetl_test4.h"

INSTANTIATE_TEST_CASE_P(FETLCylindricalKz, TestFETL,

testing::Combine(testing::Values(nTuple<3, Real>( { 0.0, 0.0, 0.0, })  //
        , nTuple<3, Real>( { -1.0, -2.0, -3.0 })

),

testing::Values(

nTuple<3, Real>( { 1.0, 2.0, 3.0 })  //
        , nTuple<3, Real>( { 2.0, 0.0, 0.0 }) //
        , nTuple<3, Real>( { 0.0, 2.0, 0.0 }) //
        , nTuple<3, Real>( { 0.0, 0.0, 2.0 }) //
        , nTuple<3, Real>( { 0.0, 2.0, 2.0 }) //
        , nTuple<3, Real>( { 2.0, 0.0, 2.0 }) //
        , nTuple<3, Real>( { 2.0, 2.0, 0.0 }) //

        ),

testing::Values(

nTuple<3, size_t>( { 1, 1, 1 }) //
        , nTuple<3, size_t>( { 17, 1, 1 }) //
        , nTuple<3, size_t>( { 1, 17, 1 }) //
        , nTuple<3, size_t>( { 1, 1, 10 }) //
        , nTuple<3, size_t>( { 1, 10, 20 }) //
        , nTuple<3, size_t>( { 17, 1, 17 }) //
        , nTuple<3, size_t>( { 17, 17, 1 }) //
        , nTuple<3, size_t>( { 12, 16, 10 })   //

        )

        ));
