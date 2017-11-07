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
#include "Shape.h"
#include "Shell.h"
namespace simpla {
namespace geometry {
struct CutCell::pimpl_s {
    std::shared_ptr<IntersectionCurveSurface> m_intersector_ = nullptr;
    std::shared_ptr<const Chart> m_chart_ = nullptr;
    //    std::shared_ptr<const Shape> m_shape_ = nullptr;
    //    Real m_tolerance_ = SP_GEO_DEFAULT_TOLERANCE;
};
CutCell::CutCell() : m_pimpl_(new pimpl_s){};
CutCell::CutCell(std::shared_ptr<const Shape> const &s, std::shared_ptr<const Chart> const &c, Real tolerance)
    : m_pimpl_(new pimpl_s) {
    ASSERT(c != nullptr)
    auto const &scale = m_pimpl_->m_chart_->GetScale();
//    tolerance = std::min(tolerance, std::sqrt(dot(scale, scale) * 0.01));
    m_pimpl_->m_chart_ = c;
    m_pimpl_->m_intersector_ = IntersectionCurveSurface::New(s, tolerance);
}
CutCell::~CutCell() { delete m_pimpl_; }
void CutCell::SetChart(std::shared_ptr<Chart> const &c) { m_pimpl_->m_chart_ = c; }
std::shared_ptr<const Chart> CutCell::GetChart() const { return m_pimpl_->m_chart_; }
void CutCell::TagCell(Array<unsigned int> *node_tags, Array<Real> *edge_tags, unsigned int tag) const {
    ASSERT(m_pimpl_->m_chart_ != nullptr);
    if (node_tags == nullptr) { return; }
    auto idx_box = node_tags->GetIndexBox();

    for (int dir = 0; dir < 3; ++dir) {
        index_tuple lo{0, 0, 0}, hi{0, 0, 0};
        std::tie(lo, hi) = idx_box;

        for (index_type i = lo[(dir + 1) % 3]; i < hi[(dir + 1) % 3]; ++i)
            for (index_type j = lo[(dir + 2) % 3]; j < hi[(dir + 2) % 3]; ++j) {
                index_tuple id{0, 0, 0};
                id[(dir + 0) % 3] = lo[dir];
                id[(dir + 1) % 3] = i;
                id[(dir + 2) % 3] = j;

                std::vector<Real> intersection_pos;
                m_pimpl_->m_intersector_->Intersect(m_pimpl_->m_chart_->GetAxis(lo, dir, hi[dir] - lo[dir]),
                                                    &intersection_pos);

                for (size_t n = 0; n < intersection_pos.size(); n += 2) {
                    auto rlo = intersection_pos[n];
                    auto rhi = intersection_pos[n + 1];
                    auto klo = static_cast<index_type>(rlo);
                    auto khi = static_cast<index_type>(rhi);

                    index_tuple s{0, 0, 0};

                    s[(dir + 1) % 3] = i;
                    s[(dir + 2) % 3] = j;
                    s[(dir + 0) % 3] = lo[dir] + klo;
                    for (s[dir] += klo + 1; s[dir] <= khi; ++s[(dir)]) {
                        node_tags->Set(node_tags->Get(s[0], s[1], s[2]) | tag, s[0], s[1], s[2]);
                    }

                    if (edge_tags != nullptr) {
                        s[(dir + 0) % 3] = lo[dir] + klo;
                        edge_tags[dir].Set(rlo - klo, s[0], s[1], s[2]);
                        s[(dir + 0) % 3] = lo[dir] + khi;
                        edge_tags[dir].Set(rhi - khi, s[0], s[1], s[2]);
                    }
                }
            }
    }
}

}  //    namespace geometry{
}  // namespace simpla{