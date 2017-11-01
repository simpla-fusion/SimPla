//
// Created by salmon on 17-10-17.
//

#include "CutCell.h"
#include "Box.h"
#include "Chart.h"
#include "Curve.h"
#include "GeoEngine.h"
#include "GeoObject.h"

namespace simpla {
namespace geometry {

CutCell::CutCell() = default;
CutCell::CutCell(std::shared_ptr<const Chart> const &c, std::shared_ptr<const Surface> const &g, Real tolerance) {
    SetUp(c, g, tolerance);
}

CutCell::~CutCell() = default;
std::shared_ptr<CutCell> CutCell::New(std::string const &s) {
    std::string key = s.empty() ? GeoEngine::RegisterName_s() : s;
    auto res = Factory<CutCell>::Create(key);
    if (res == nullptr) {
        RUNTIME_ERROR << "Create CutCell Fail! [" << key << "]" << std::endl << CutCell::ShowDescription();
    }
    return res;
}
std::shared_ptr<CutCell> CutCell::New(std::shared_ptr<data::DataNode> const &d) {
    std::shared_ptr<CutCell> res = nullptr;
    if (d != nullptr) {
        res = New(d->GetValue<std::string>("_TYPE_", ""));
        res->Deserialize(d);
    }
    if (res == nullptr) { RUNTIME_ERROR << "Create CutCell Fail! " << *d << std::endl << CutCell::ShowDescription(); }
    return res;
}

void CutCell::SetUp(std::shared_ptr<const Chart> const &c, std::shared_ptr<const Surface> const &s, Real tolerance) {
    m_chart_ = c;
    m_surface_ = s;
    m_tolerance_ = tolerance;
}

void CutCell::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {}
std::shared_ptr<simpla::data::DataNode> CutCell::Serialize() const {
    return data::DataNode::New(data::DataNode::DN_TABLE);
}
size_type CutCell::IntersectAxe(index_tuple const &idx, int dir, index_type length, std::vector<Real> *u) const {
    auto res = m_surface_->GetIntersection(m_chart_->GetAxis(idx, dir, length), m_tolerance_);

    return 0;
}

void CutCell::TagCell(Array<unsigned int> *vertex_tags, Array<Real> *edge_tags, unsigned int tag) const {
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
    //    for (int dir = 0; dir < 3; ++dir) {
    //        index_tuple lo{0, 0, 0}, hi{0, 0, 0};
    //        std::tie(lo, hi) = idx_box;
    //
    //        for (index_type i = lo[(dir + 1) % 3]; i < hi[(dir + 1) % 3]; ++i)
    //            for (index_type j = lo[(dir + 2) % 3]; j < hi[(dir + 2) % 3]; ++j) {
    //                index_tuple id;
    //                id[(dir + 0) % 3] = lo[dir];
    //                id[(dir + 1) % 3] = i;
    //                id[(dir + 2) % 3] = j;
    //                point_type x_begin = chart->uvw(id[0], id[1], id[2]);
    //                id[(dir + 0) % 3] = hi[dir];
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
    //                    s[(dir + 1) % 3] = i;
    //                    s[(dir + 2) % 3] = j;
    //                    if (edge_tags != nullptr) {
    //                        s[(dir + 0) % 3] = lo[dir] + klo;
    //                        edge_tags[dir].Set(rlo - klo, s[0], s[1], s[2]);
    //                        s[(dir + 0) % 3] = lo[dir] + khi;
    //                        edge_tags[dir].Set(rhi - khi, s[0], s[1], s[2]);
    //                    }
    //                    if (node_tags != nullptr) {
    //                        s[(dir + 0) % 3] = lo[dir] + klo;
    //                        for (s[dir] += klo + 1; s[dir] <= khi; ++s[(dir)]) {
    //                            node_tags->Set(node_tags->Get(s[0], s[1], s[2]) | tag, s[0], s[1], s[2]);
    //                        }
    //                    }
    //                }
    //            }
    //    }
}

}  //    namespace geometry{
}  // namespace simpla{