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
#include "Intersector.h"

namespace simpla {
namespace geometry {

void CutCellTagNode(Array<unsigned int> *node_tags, Array<Real> *edge_tags, std::shared_ptr<const Chart> const &chart,
                    index_box_type const &idx_box, const std::shared_ptr<const GeoObject> &g, unsigned int tag) {
//    if (auto box = std::dynamic_pointer_cast<const Box>(g)) {
//        auto bound_box = box->GetBoundingBox();
//        index_tuple lo, hi;
//        lo = std::get<1>(chart->invert_local_coordinates(std::get<0>(bound_box)));
//        hi = std::get<1>(chart->invert_local_coordinates(std::get<1>(bound_box)));
//        auto a_selection = node_tags->GetSelection(std::make_tuple(lo, hi));
//        a_selection |= tag;
//        return;
//    }

    auto const &scale = chart->GetScale();
    Real tolerance = std::sqrt(dot(scale, scale) * 0.01);
    point_type xlo, xhi;
    std::tie(xlo, xhi) = g->GetBoundingBox();
    vector_type length = xhi - xlo;
    xlo -= 0.03 * length;
    xhi += 0.03 * length;
    auto m_body_inter_ = Intersector::New(g, tolerance);

    for (int dir = 0; dir < 3; ++dir) {
        index_tuple lo{0, 0, 0}, hi{0, 0, 0};
        std::tie(lo, hi) = idx_box;

        for (index_type i = lo[(dir + 1) % 3]; i < hi[(dir + 1) % 3]; ++i)
            for (index_type j = lo[(dir + 2) % 3]; j < hi[(dir + 2) % 3]; ++j) {
                index_tuple id;
                id[(dir + 0) % 3] = lo[dir];
                id[(dir + 1) % 3] = i;
                id[(dir + 2) % 3] = j;
                point_type x_begin = chart->global_coordinates(0b0, id[0], id[1], id[2]);
                std::vector<Real> intersection_pos;
                m_body_inter_->GetIntersectionPoints(chart->GetAxis(x_begin, dir), intersection_pos);

                for (size_t n = 0; n < intersection_pos.size(); n += 2) {
                    auto rlo = intersection_pos[n];
                    auto rhi = intersection_pos[n + 1];
                    auto klo = static_cast<index_type>(rlo);
                    auto khi = static_cast<index_type>(rhi);

                    index_tuple s;

                    s[(dir + 1) % 3] = i;
                    s[(dir + 2) % 3] = j;
                    if (edge_tags != nullptr) {
                        s[(dir + 0) % 3] = lo[dir] + klo;
                        edge_tags[dir].Set(rlo - klo, s[0], s[1], s[2]);
                        s[(dir + 0) % 3] = lo[dir] + khi;
                        edge_tags[dir].Set(rhi - khi, s[0], s[1], s[2]);
                    }
                    if (node_tags != nullptr) {
                        s[(dir + 0) % 3] = lo[dir] + klo;
                        for (s[dir] += klo + 1; s[dir] <= khi; ++s[(dir)]) {
                            node_tags->Set(node_tags->Get(s[0], s[1], s[2]) | tag, s[0], s[1], s[2]);
                        }
                    }
                }
            }
    }
}

}  //    namespace geometry{

}  // namespace simpla{