//
// Created by salmon on 16-11-24.
//
#include "Mesh.h"
#include <simpla/algebra/EntityId.h>
#include <simpla/engine/Attribute.h>
#include <simpla/engine/MeshBlock.h>
#include <simpla/engine/Patch.h>
#include <simpla/geometry/Chart.h>
#include <simpla/geometry/GeoObject.h>
namespace simpla {
namespace mesh {

struct MeshBase::pimpl_s {
    engine::MeshBlock m_mesh_block_;
    std::shared_ptr<geometry::Chart> m_chart_;
    point_type m_origin_ = {1, 1, 1};
    point_type m_coarsest_cell_width_ = {1, 1, 1};
    index_tuple m_ghost_width_{2, 2, 2};
    index_tuple m_idx_origin_{0, 0, 0};
    index_tuple m_dimensions_{1, 1, 1};
    size_tuple m_periodic_dimension_{0, 0, 0};
};
MeshBase::MeshBase(std::shared_ptr<geometry::Chart> const& c, std::string const& s_name) : m_pimpl_(new pimpl_s) {
    SetChart(c);
}
MeshBase::~MeshBase() {}
void MeshBase::SetChart(std::shared_ptr<geometry::Chart> const& c) { m_pimpl_->m_chart_ = c; }
std::shared_ptr<geometry::Chart> MeshBase::GetChart() const { return m_pimpl_->m_chart_; }

point_type const& MeshBase::GetCellWidth() const { return m_pimpl_->m_coarsest_cell_width_; }
point_type const& MeshBase::GetOrigin() const { return m_pimpl_->m_origin_; }

void MeshBase::SetPeriodicDimension(size_tuple const& x) { m_pimpl_->m_periodic_dimension_ = x; }
size_tuple const& MeshBase::GetPeriodicDimension() const { return m_pimpl_->m_periodic_dimension_; }

void MeshBase::SetDefaultGhostWidth(index_tuple const& g) { m_pimpl_->m_ghost_width_ = g; }
index_tuple MeshBase::GetDefaultGhostWidth() const { return m_pimpl_->m_ghost_width_; }

void MeshBase::FitBoundBox(box_type const& b) {
    m_pimpl_->m_coarsest_cell_width_ = (std::get<1>(b) - std::get<0>(b)) / m_pimpl_->m_dimensions_;
    m_pimpl_->m_origin_ = std::get<0>(b) - m_pimpl_->m_idx_origin_ * m_pimpl_->m_coarsest_cell_width_;
}

void MeshBase::SetDimensions(index_tuple const& d) { m_pimpl_->m_dimensions_ = d; }
index_tuple MeshBase::GetDimensions() const { return m_pimpl_->m_dimensions_; }
index_tuple MeshBase::GetIndexOffset() const { return m_pimpl_->m_idx_origin_; }

void MeshBase::SetBlock(const engine::MeshBlock& m) { m_pimpl_->m_mesh_block_ = m; }
const engine::MeshBlock& MeshBase::GetBlock() const { return m_pimpl_->m_mesh_block_; }

id_type MeshBase::GetBlockId() const { return m_pimpl_->m_mesh_block_.GetGUID(); }

std::shared_ptr<data::DataTable> MeshBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    //    p->SetValue("Type", GetRegisterName());

    if (m_pimpl_->m_chart_ != nullptr) { p->SetValue("Chart", m_pimpl_->m_chart_->Serialize()); }

    p->SetValue("PeriodicDimensions", m_pimpl_->m_periodic_dimension_);
    p->SetValue("Dimensions", m_pimpl_->m_dimensions_);
    p->SetValue("IndexOrigin", m_pimpl_->m_idx_origin_);
    p->SetValue("CoarsestCellWidth", m_pimpl_->m_coarsest_cell_width_);
    return p;
}
void MeshBase::Deserialize(const std::shared_ptr<data::DataTable>& cfg) {
    m_pimpl_->m_idx_origin_ = cfg->GetValue<nTuple<int, 3>>("IndexOrigin", nTuple<int, 3>{0, 0, 0});
    m_pimpl_->m_dimensions_ = cfg->GetValue<nTuple<int, 3>>("Dimensions", nTuple<int, 3>{1, 1, 1});
    m_pimpl_->m_periodic_dimension_ = cfg->GetValue<nTuple<int, 3>>("PeriodicDimension", nTuple<int, 3>{0, 0, 0});
}
index_tuple MeshBase::GetGhostWidth(int tag) const { return GetBlock().GetGhostWidth(); }

box_type MeshBase::GetBox() const {
    box_type res;
    index_tuple lo, hi;
    std::tie(lo, hi) = GetIndexBox(VERTEX);
    std::get<0>(res) = point(
        EntityId{static_cast<int16_t>(lo[0]), static_cast<int16_t>(lo[1]), static_cast<int16_t>(lo[2]), 0}, nullptr);
    std::get<1>(res) = point(
        EntityId{static_cast<int16_t>(hi[0]), static_cast<int16_t>(hi[1]), static_cast<int16_t>(hi[2]), 0}, nullptr);
    return res;
}

point_type MeshBase::map(point_type const& p) const { return m_pimpl_->m_chart_->map(p); }

// index_box_type MeshBase::GetIndexBox(int tag) const {
//    index_box_type res = GetBlock()->GetIndexBox();
//    switch (tag) {
//        case 0:
//            std::get<1>(res) += 1;
//            break;
//        case 1:
//            std::get<1>(res)[1] += 1;
//            std::get<1>(res)[2] += 1;
//            break;
//        case 2:
//            std::get<1>(res)[0] += 1;
//            std::get<1>(res)[2] += 1;
//            break;
//        case 4:
//            std::get<1>(res)[0] += 1;
//            std::get<1>(res)[1] += 1;
//            break;
//        case 3:
//            std::get<1>(res)[2] += 1;
//            break;
//        case 5:
//            std::get<1>(res)[1] += 1;
//            break;
//        case 6:
//            std::get<1>(res)[0] += 1;
//            break;
//        case 7:
//        default:
//            break;
//    }
//    return res;
//}
//
// EntityRange MeshBase::GetRange(std::string const& k) const {
//    if (m_pimpl_->m_ranges_ == nullptr) {
//        return EntityRange{};
//    } else {
//        auto it = m_pimpl_->m_ranges_->find(k);
//        return (it != m_pimpl_->m_ranges_->end()) ? it->second : EntityRange{};
//    }
//};
//
// EntityRange MeshBase::GetBodyRange(int IFORM, std::string const& k) const {
//    return GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + "_BODY");
//};
// EntityRange MeshBase::GetBoundaryRange(int IFORM, std::string const& k, bool is_parallel) const {
//    auto res =
//        (IFORM == VERTEX || IFORM == VOLUME)
//            ? GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + "_BOUNDARY")
//            : GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + (is_parallel ? "_PARA" : "_PERP") +
//            "_BOUNDARY");
//    return res;
//};
// EntityRange MeshBase::GetParallelBoundaryRange(int IFORM, std::string const& k) const {
//    return GetBoundaryRange(IFORM, k, true);
//}
// EntityRange MeshBase::GetPerpendicularBoundaryRange(int IFORM, std::string const& k) const {
//    return GetBoundaryRange(IFORM, k, false);
//}
// EntityRange MeshBase::GetGhostRange(int IFORM) const {
//    return GetRange("." + std::string(EntityIFORMName[IFORM]) + "_GHOST");
//}
// EntityRange MeshBase::GetInnerRange(int IFORM) const {
//    return GetRange("." + std::string(EntityIFORMName[IFORM]) + "_INNER");
//}
// std::shared_ptr<std::map<std::string, EntityRange>> MeshBase::GetRanges() { return m_pimpl_->m_ranges_; };
// std::shared_ptr<std::map<std::string, EntityRange>> MeshBase::GetRanges() const { return m_pimpl_->m_ranges_; };
// void MeshBase::SetRanges(std::shared_ptr<std::map<std::string, EntityRange>> const& r) {
//    m_pimpl_->m_ranges_ = r;
//    Click();
//};
}  // namespace mesh
}  // namespace simpla
