/**
 * @file metric_test.cpp.cpp
 * @author salmon
 * @date 2015-11-04.
 */

#include <gtest/gtest.h>

#include "simpla/algebra/nTuple.h"
#include "simpla/algebra/nTupleExt.h"
#include "../../toolbox/utilities/Log.h"
#include "../../toolbox/iterator/range.h"

#include "../../toolbox/utilities/Log.h"


#include "simpla/geometry/csCartesian.h"
#include "simpla/geometry/csCylindrical.h"
#include "Constants.h"

using namespace simpla;

TEST(MetricTest, volume)
{
    Metric<coordinate_system::Cylindrical<>> metric;

    Real dR = 1;
    Real R0 = 1.0;
    Real dPhi = 2.0;
    Real dZ = 0.2;



    /**
     *\verbatim
     *                ^Z
     *               /
     *        Phi   /
     *        ^    /
     *        |   6---------------7
     *        |  /|              /|
     *        | / |             / |
     *        |/  |            /  |
     *        4---|-----------5   |
     *        |   |           |   |
     *        |   2-----------|---3
     *        |  /            |  /
     *        | /             | /
     *        |/              |/
     *0------R0---------------1---> R
     *
     *\endverbatim
     */

    nTuple<Real, 3> p[8] = {

            {R0,      0,  0},/*000*/

            {R0 + dR, 0,  0},/*001*/

            {R0,      dZ, 0},/*010*/

            {R0 + dR, dZ, 0},/*011*/

            {R0,      0,  dPhi},/*100*/

            {R0 + dR, 0,  dPhi},/*101*/

            {R0,      dZ, dPhi},/*110*/

            {R0 + dR, dZ, dPhi}/*111*/

    };

    //EDGE
    EXPECT_DOUBLE_EQ(dR, metric.simplex_length(p[0], p[1]));
    EXPECT_DOUBLE_EQ(dR, metric.simplex_length(p[2], p[3]));
    EXPECT_DOUBLE_EQ(dR, metric.simplex_length(p[4], p[5]));
    EXPECT_DOUBLE_EQ(dR, metric.simplex_length(p[6], p[7]));


    EXPECT_DOUBLE_EQ(dZ, metric.simplex_length(p[0], p[2]));
    EXPECT_DOUBLE_EQ(dZ, metric.simplex_length(p[1], p[3]));
    EXPECT_DOUBLE_EQ(dZ, metric.simplex_length(p[4], p[6]));
    EXPECT_DOUBLE_EQ(dZ, metric.simplex_length(p[5], p[7]));


    EXPECT_DOUBLE_EQ(R0 * dPhi, metric.simplex_length(p[0], p[4]));
    EXPECT_DOUBLE_EQ(R0 * dPhi, metric.simplex_length(p[2], p[6]));
    EXPECT_DOUBLE_EQ((R0 + dR) * dPhi, metric.simplex_length(p[1], p[5]));
    EXPECT_DOUBLE_EQ((R0 + dR) * dPhi, metric.simplex_length(p[3], p[7]));

    // FACE
    EXPECT_DOUBLE_EQ(R0 * dPhi * dZ,
                     metric.simplex_area(p[0], p[4], p[6]) + metric.simplex_area(p[0], p[6], p[2]));


    EXPECT_DOUBLE_EQ(0.5 * dPhi * dR * (2.0 * R0 + dR),
                     metric.simplex_area(p[0], p[1], p[5]) + metric.simplex_area(p[0], p[5], p[4]));


    EXPECT_DOUBLE_EQ(dR * dZ,
                     metric.simplex_area(p[0], p[1], p[3]) + metric.simplex_area(p[0], p[3], p[2]));



    EXPECT_DOUBLE_EQ((R0 + dR) * dPhi * dZ,
                     metric.simplex_area(p[1], p[3], p[7]) + metric.simplex_area(p[1], p[7], p[5]));


    EXPECT_DOUBLE_EQ(0.5 * dPhi * dR * (2.0 * R0 + dR),
                     metric.simplex_area(p[2], p[7], p[3]) + metric.simplex_area(p[2], p[6], p[7]));


    EXPECT_DOUBLE_EQ(dR * dZ,
                     metric.simplex_area(p[4], p[5], p[7]) + metric.simplex_area(p[4], p[7], p[6]));


    CHECK(0.5 * dPhi * dR * (2.0 * R0 + dR) * dZ);

    EXPECT_DOUBLE_EQ(0.5 * dPhi * dR * (2.0 * R0 + dR) * dZ,
                     metric.simplex_volume(p[0], p[1], p[2], p[4]) + //
                     metric.simplex_volume(p[1], p[4], p[5], p[2]) + //
                     metric.simplex_volume(p[2], p[6], p[4], p[5]) + //
                     metric.simplex_volume(p[1], p[3], p[2], p[5]) + //
                     metric.simplex_volume(p[3], p[5], p[7], p[6]) + //
                     metric.simplex_volume(p[3], p[6], p[2], p[5])

    );


}