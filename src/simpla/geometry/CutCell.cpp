//
// Created by salmon on 17-10-17.
//
#ifdef OCE_FOUND
#include "occ/GeoObjectOCC.h"
#include "occ/OCECutCell.h"
#endif

#include "Box.h"
#include "Chart.h"
#include "Curve.h"
#include "CutCell.h"
#include "GeoObject.h"
#include "IntersectionCurveSurface.h"

namespace simpla {
namespace geometry {

void CutCellTagNode(Array<unsigned int> *node_tags, std::shared_ptr<const Chart> const &chart,
                    index_box_type const &idx_box, const std::shared_ptr<const GeoObject> &g, unsigned int tag) {
    //    if (auto box = std::dynamic_pointer_cast<const Box>(g)) {
    //        auto bound_box = box->GetBoundingBox();
    //        index_tuple lo, hi;
    //        lo = std::get<1>(chart->invert_local_coordinates(std::get<0>(bound_box)));
    //        hi = std::get<1>(chart->invert_local_coordinates(std::get<1>(bound_box)));
    //        auto a_selection = node_tags->GetSelection(std::make_tuple(lo, hi));
    //        a_selection |= tag;
    //    }
    auto const &scale = chart->GetScale();
    Real tolerance = std::sqrt(dot(scale, scale) * 0.01);
    point_type xlo, xhi;
    std::tie(xlo, xhi) = g->GetBoundingBox();
    vector_type length = xhi - xlo;
    xlo -= 0.03 * length;
    xhi += 0.03 * length;

    IntersectionCurveSurface m_body_inter_;
    m_body_inter_.Load(g, tolerance);

    for (int dir = 0; dir < 3; ++dir) {
        index_tuple lo{0, 0, 0}, hi{0, 0, 0};
        std::tie(lo, hi) = idx_box;
        hi[dir] = lo[dir] + 1;
        for (index_type i = lo[0]; i < hi[0]; ++i)
            for (index_type j = lo[1]; j < hi[1]; ++j)
                for (index_type k = lo[2]; k < hi[2]; ++k) {
                    point_type x_begin = chart->global_coordinates(0b0, i, j, k);
                    auto curve = chart->GetAxisCurve(x_begin, dir);
                    std::vector<Real> intersection_points;
                    m_body_inter_.Intersect(curve, intersection_points);

                    for (size_t n = 0; n < intersection_points.size(); n += 2) {
                        auto p0 = curve->Value(intersection_points[n]);
                        auto p1 = curve->Value(intersection_points[n + 1]);

                        index_tuple i0, i1;
                        point_type r0, r1;
                        std::tie(r0, i0) = chart->invert_global_coordinates(p0);
                        std::tie(r1, i1) = chart->invert_global_coordinates(p1);

                        index_tuple id{i, j, k};
                        for (id[dir] = i0[dir]; id[dir] <= i1[dir]; ++id[dir]) {
                            node_tags->Set(node_tags->Get(id[0], id[1], id[2]) | tag, id[0], id[1], id[2]);
                        }
                    }
                }
    }
}

}  //    namespace geometry{

}  // namespace simpla{