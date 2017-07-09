//
// Created by salmon on 16-11-24.
//
#include "Mesh.h"
#include <simpla/algebra/EntityId.h>
#include <simpla/engine/Attribute.h>
#include <simpla/engine/MeshBlock.h>
#include <simpla/engine/Patch.h>
#include <simpla/model/Chart.h>
#include <simpla/model/GeoObject.h>
//#include <simpla/model/Model.h>
namespace simpla {

struct MeshBase::pimpl_s {
    engine::MeshBlock m_mesh_block_;
    std::shared_ptr<model::Chart> m_chart_;
    point_type m_origin_ = {1, 1, 1};
    point_type m_coarsest_cell_width_ = {1, 1, 1};
    index_tuple m_ghost_width_{2, 2, 2};
    index_tuple m_idx_origin_{0, 0, 0};
    index_tuple m_dimensions_{1, 1, 1};
    size_tuple m_periodic_dimension_{0, 0, 0};
};
MeshBase::MeshBase(std::shared_ptr<model::Chart> const& c, std::string const& s_name)
    : SPObject(s_name), m_pimpl_(new pimpl_s) {
    SetChart(c);
}
MeshBase::~MeshBase() {}
void MeshBase::SetChart(std::shared_ptr<model::Chart> const& c) {
    m_pimpl_->m_chart_ = c;
    Click();
}
std::shared_ptr<model::Chart> MeshBase::GetChart() const { return m_pimpl_->m_chart_; }

MeshBase const* MeshBase::GetMesh() const { return this; }

point_type const& MeshBase::GetCellWidth() const { return m_pimpl_->m_coarsest_cell_width_; }
point_type const& MeshBase::GetOrigin() const { return m_pimpl_->m_origin_; }

void MeshBase::SetPeriodicDimension(size_tuple const& x) {
    m_pimpl_->m_periodic_dimension_ = x;
    Click();
}
size_tuple const& MeshBase::GetPeriodicDimension() const { return m_pimpl_->m_periodic_dimension_; }

void MeshBase::SetDefaultGhostWidth(index_tuple const& g) {
    m_pimpl_->m_ghost_width_ = g;
    Click();
}
index_tuple MeshBase::GetDefaultGhostWidth() const { return m_pimpl_->m_ghost_width_; }

void MeshBase::FitBoundBox(box_type const& b) {
    m_pimpl_->m_coarsest_cell_width_ = (std::get<1>(b) - std::get<0>(b)) / m_pimpl_->m_dimensions_;
    m_pimpl_->m_origin_ = std::get<0>(b) - m_pimpl_->m_idx_origin_ * m_pimpl_->m_coarsest_cell_width_;
    Click();
}

void MeshBase::SetDimensions(index_tuple const& d) { m_pimpl_->m_dimensions_ = d; }
index_tuple MeshBase::GetDimensions() const { return m_pimpl_->m_dimensions_; }
index_tuple MeshBase::GetIndexOffset() const { return m_pimpl_->m_idx_origin_; }

void MeshBase::DoUpdate() { engine::SPObject::DoUpdate(); };

void MeshBase::SetBlock(const engine::MeshBlock& m) {
    m_pimpl_->m_mesh_block_ = m;
    Click();
}
const engine::MeshBlock& MeshBase::GetBlock() const { return m_pimpl_->m_mesh_block_; }

id_type MeshBase::GetBlockId() const { return m_pimpl_->m_mesh_block_.GetGUID(); }

void MeshBase::InitializeData(Real time_now) { Update(); }
void MeshBase::SetBoundaryCondition(Real time_now, Real time_dt) { Update(); }

std::shared_ptr<data::DataTable> MeshBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());

    if (m_pimpl_->m_chart_ != nullptr) { p->SetValue("Chart", m_pimpl_->m_chart_->Serialize()); }

    p->SetValue("PeriodicDimensions", m_pimpl_->m_periodic_dimension_);
    p->SetValue("Dimensions", m_pimpl_->m_dimensions_);
    p->SetValue("IndexOrigin", m_pimpl_->m_idx_origin_);
    p->SetValue("CoarsestCellWidth", m_pimpl_->m_coarsest_cell_width_);
    return p;
}
void MeshBase::Deserialize(const std::shared_ptr<data::DataTable>& cfg) {
    m_pimpl_->m_chart_ = cfg->has("Coordinates") ? model::Chart::Create(cfg->Get("Coordinates"))
                                                 : model::Chart::Create("Cartesian");
    m_pimpl_->m_idx_origin_ = cfg->GetValue<nTuple<int, 3>>("IndexOrigin", nTuple<int, 3>{0, 0, 0});
    m_pimpl_->m_dimensions_ = cfg->GetValue<nTuple<int, 3>>("Dimensions", nTuple<int, 3>{1, 1, 1});
    m_pimpl_->m_periodic_dimension_ = cfg->GetValue<nTuple<int, 3>>("PeriodicDimension", nTuple<int, 3>{0, 0, 0});
    Update();
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

void MeshBase::Push(engine::Patch* p) {
    Click();
    SetBlock(p->GetBlock());
    engine::AttributeGroup::Push(p);
    DoUpdate();
}
void MeshBase::Pull(engine::Patch* p) {
    p->SetBlock(GetBlock());
    engine::AttributeGroup::Pull(p);
    Click();
    DoTearDown();
}

// index_box_type Mesh::GetIndexBox(int tag) const {
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
// EntityRange Mesh::GetRange(std::string const& k) const {
//    if (m_pimpl_->m_ranges_ == nullptr) {
//        return EntityRange{};
//    } else {
//        auto it = m_pimpl_->m_ranges_->find(k);
//        return (it != m_pimpl_->m_ranges_->end()) ? it->second : EntityRange{};
//    }
//};
//
// EntityRange Mesh::GetBodyRange(int IFORM, std::string const& k) const {
//    return GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + "_BODY");
//};
// EntityRange Mesh::GetBoundaryRange(int IFORM, std::string const& k, bool is_parallel) const {
//    auto res =
//        (IFORM == VERTEX || IFORM == VOLUME)
//            ? GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + "_BOUNDARY")
//            : GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + (is_parallel ? "_PARA" : "_PERP") +
//            "_BOUNDARY");
//    return res;
//};
// EntityRange Mesh::GetParallelBoundaryRange(int IFORM, std::string const& k) const {
//    return GetBoundaryRange(IFORM, k, true);
//}
// EntityRange Mesh::GetPerpendicularBoundaryRange(int IFORM, std::string const& k) const {
//    return GetBoundaryRange(IFORM, k, false);
//}
// EntityRange Mesh::GetGhostRange(int IFORM) const {
//    return GetRange("." + std::string(EntityIFORMName[IFORM]) + "_GHOST");
//}
// EntityRange Mesh::GetInnerRange(int IFORM) const {
//    return GetRange("." + std::string(EntityIFORMName[IFORM]) + "_INNER");
//}
// std::shared_ptr<std::map<std::string, EntityRange>> Mesh::GetRanges() { return m_pimpl_->m_ranges_; };
// std::shared_ptr<std::map<std::string, EntityRange>> Mesh::GetRanges() const { return m_pimpl_->m_ranges_; };
// void Mesh::SetRanges(std::shared_ptr<std::map<std::string, EntityRange>> const& r) {
//    m_pimpl_->m_ranges_ = r;
//    Click();
//};
}  // namespace simpla
