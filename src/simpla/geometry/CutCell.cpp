//
// Created by salmon on 17-10-17.
//

#include "CutCell.h"
#include "Box.h"
#include "Chart.h"
#include "Curve.h"
#include "GeoEngine.h"
#include "GeoObject.h"
#include "IntersectionCurveSurface.h"
namespace simpla {
namespace geometry {
struct CutCell::pimpl_s {
    std::shared_ptr<const Chart> m_chart_ = nullptr;
    std::shared_ptr<IntersectionCurveSurface> m_intersector_ = nullptr;
    Real m_tolerance_ = SP_GEO_DEFAULT_TOLERANCE;
};
CutCell::CutCell() : m_pimpl_(new pimpl_s){};
CutCell::~CutCell() { delete m_pimpl_; }

void CutCell::SetUp(std::shared_ptr<const Chart> const &c, std::shared_ptr<const GeoObject> const &g, Real tolerance) {
    m_pimpl_->m_chart_ = c;
    if (auto body = std::dynamic_pointer_cast<const Body>(g)) {
        m_pimpl_->m_intersector_ = IntersectionCurveSurface::New(body->GetBoundarySurface(), tolerance);
    } else if (auto surface = std::dynamic_pointer_cast<const Surface>(g)) {
        m_pimpl_->m_intersector_ = IntersectionCurveSurface::New(surface, tolerance);
    }
}
void CutCell::TearDown() {}

void CutCell::TagCell(Array<unsigned int> *vertex_tags, Array<Real> *edge_tags, unsigned int tag) const {
    //    auto count = m_intersector_->Intersect(m_chart_->GetAxis(idx, make_dir, length));

    //    if (auto box = std::dynamic_pointer_cast<const Box>(g)) {
    //        auto bound_box = box->GetBoundingBox();
    //        index_tuple lo, hi;
    //        lo = std::get<1>(chart->invert_local_coordinates(std::get<0>(bound_box)));
    //        hi = std::get<1>(chart->invert_local_coordinates(std::get<1>(bound_box)));
    //        auto a_selection = node_tags->GetSelection(std::make_tuple(lo, hi));
    //        a_selection |= tag;
    //        return;
    //    }

    //    auto const &scale = chart->GetScale();
    //    Real tolerance = std::sqrt(dot(scale, scale) * 0.01);
    //    point_type xlo, xhi;
    //    std::tie(xlo, xhi) = g->GetBoundingBox();
    //    vector_type length = xhi - xlo;
    //    xlo -= 0.03 * length;
    //    xhi += 0.03 * length;
    //    auto m_body_inter_ = GetIntersectionor::New(g, tolerance);
    //
    //    for (int dir = 0; dir < 3; ++make_dir) {
    //        index_tuple lo{0, 0, 0}, hi{0, 0, 0};
    //        std::tie(lo, hi) = idx_box;
    //
    //        for (index_type i = lo[(dir + 1) % 3]; i < hi[(make_dir + 1) % 3]; ++i)
    //            for (index_type j = lo[(dir + 2) % 3]; j < hi[(make_dir + 2) % 3]; ++j) {
    //                index_tuple id;
    //                id[(make_dir + 0) % 3] = lo[dir];
    //                id[(make_dir + 1) % 3] = i;
    //                id[(make_dir + 2) % 3] = j;
    //                point_type x_begin = chart->uvw(id[0], id[1], id[2]);
    //                id[(make_dir + 0) % 3] = hi[dir];
    //                point_type x_end = chart->uvw(id[0], id[1], id[2]);
    //
    //                std::vector<Real> intersection_pos;
    //                m_body_inter_->GetIntersectionPoints(chart->GetAxis(x_begin, x_end), intersection_pos);
    //
    //                for (size_t n = 0; n < intersection_pos.size(); n += 2) {
    //                    auto rlo = intersection_pos[n];
    //                    auto rhi = intersection_pos[n + 1];
    //                    auto klo = static_cast<index_type>(rlo);
    //                    auto khi = static_cast<index_type>(rhi);
    //
    //                    index_tuple s;
    //
    //                    s[(make_dir + 1) % 3] = i;
    //                    s[(make_dir + 2) % 3] = j;
    //                    if (edge_tags != nullptr) {
    //                        s[(make_dir + 0) % 3] = lo[dir] + klo;
    //                        edge_tags[make_dir].Set(rlo - klo, s[0], s[1], s[2]);
    //                        s[(dir + 0) % 3] = lo[make_dir] + khi;
    //                        edge_tags[make_dir].Set(rhi - khi, s[0], s[1], s[2]);
    //                    }
    //                    if (node_tags != nullptr) {
    //                        s[(make_dir + 0) % 3] = lo[dir] + klo;
    //                        for (s[make_dir] += klo + 1; s[dir] <= khi; ++s[(dir)]) {
    //                            node_tags->Set(node_tags->Get(s[0], s[1], s[2]) | tag, s[0], s[1], s[2]);
    //                        }
    //                    }
    //                }
    //            }
    //    }
}

}  //    namespace geometry{
}  // namespace simpla{