//
// Created by salmon on 17-10-30.
//

#include "GeoAlgorithm.h"
#include <vector>
namespace simpla {
namespace geometry {

bool TestPointInBox(nTuple<Real, 2> const& p, nTuple<Real, 2> const& bMin, nTuple<Real, 2> const& bMax) {
    return bMin[0] <= p[0] && p[0] <= bMax[0] &&  //
           bMin[1] <= p[1] && p[1] <= bMax[1];
}
bool TestPointInBox(nTuple<Real, 2> const& p, std::tuple<nTuple<Real, 2>, nTuple<Real, 2>> const& box) {
    return TestPointInBox(p, std::get<0>(box), std::get<1>(box));
}
bool TestPointInBox(point_type const& p, point_type const& bMin, point_type const& bMax) {
    return bMin[0] <= p[0] && p[0] <= bMax[0] &&  //
           bMin[1] <= p[1] && p[1] <= bMax[1] &&  //
           bMin[2] <= p[2] && p[2] <= bMax[2];
}
bool TestPointInBox(point_type const& p, std::tuple<point_type, point_type> const& box) {
    return TestPointInBox(p, std::get<0>(box), std::get<1>(box));
}
bool TestPointOnPlane(point_type const& p, point_type const& o, vector_type const& normal, Real tolerance) {
    return std::abs(dot(p - o, normal)) < tolerance;
}

bool TestIntersectionCubeSphere(point_type const& bMin, point_type const& bMax, point_type const& C, Real r) {
    auto r2 = r * r;
    for (int i = 0; i < 3; i++) {
        if (C[i] < bMin[i])
            r2 -= sp_sqrt(C[i] - bMin[i]);
        else if (C[i] > bMax[i])
            r2 -= sp_sqrt(C[i] - bMax[i]);
    }
    return r2 > 0;
}
int IntersectLineSphere(point_type const& p0, point_type const& p1, point_type const& c, Real r, Real tolerance,
                        std::vector<Real>& res) {
    int count = 0;
    point_type const& o = p0;
    vector_type l = p1 - p0;
    auto l_oc = dot(l, o - c);
    auto oc = dot(o - c, o - c);
    auto t = l_oc * l_oc - oc * oc + r * r;
    if (t < 0) {
    } else if (std::abs(t) < tolerance) {  // one point
        res.push_back(-l_oc + t);
        count = 1;
    } else {  // two point
        res.push_back(-l_oc + t);
        res.push_back(-l_oc - t);
        count = 2;
    }
    return count;
}

}  // namespace geometry
}  // namespace simpla{
