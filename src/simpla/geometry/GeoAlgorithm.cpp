//
// Created by salmon on 17-10-30.
//

#include "GeoAlgorithm.h"
#include <vector>
namespace simpla {
namespace geometry {

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
