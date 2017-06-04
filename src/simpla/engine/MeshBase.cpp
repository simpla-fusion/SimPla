//
// Created by salmon on 16-11-24.
//
#include "MeshBase.h"
#include <simpla/geometry/Chart.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/utilities/EntityId.h>
#include "Attribute.h"
#include "Domain.h"
#include "MeshBlock.h"
#include "Patch.h"
#include "simpla/model/Model.h"
namespace simpla {
namespace engine {

struct MeshBase::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_block_;
    std::shared_ptr<geometry::Chart> m_chart_;

    point_type m_origin_ = {1, 1, 1};
    point_type m_coarsest_cell_width_ = {1, 1, 1};
    index_tuple m_ghost_width_{2, 2, 2};
    index_tuple m_idx_origin_{0, 0, 0};
    index_tuple m_dimensions_{1, 1, 1};
    size_tuple m_periodic_dimension_{0, 0, 0};
    Patch* m_patch_ = nullptr;
};
MeshBase::MeshBase(std::shared_ptr<geometry::Chart> const& c, std::string const& s_name)
    : SPObject(s_name), m_pimpl_(new pimpl_s) {
    SetChart(c);
}
MeshBase::~MeshBase() {}
void MeshBase::SetChart(std::shared_ptr<geometry::Chart> const& c) {
    m_pimpl_->m_chart_ = c;
    Click();
}
std::shared_ptr<geometry::Chart> MeshBase::GetChart() const { return m_pimpl_->m_chart_; }

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

void MeshBase::Update() { Tag(); };

void MeshBase::SetBlock(std::shared_ptr<MeshBlock> m) {
    m_pimpl_->m_mesh_block_ = m;
    Click();
}
std::shared_ptr<MeshBlock> MeshBase::GetBlock() const { return m_pimpl_->m_mesh_block_; }

id_type MeshBase::GetBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->GetGUID();
}

void MeshBase::InitializeData(Real time_now) { DoUpdate(); }
void MeshBase::SetBoundaryCondition(Real time_now, Real time_dt) { DoUpdate(); }

std::shared_ptr<data::DataTable> MeshBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());

    if (m_pimpl_->m_chart_ != nullptr) { p->SetValue("Chart", m_pimpl_->m_chart_->Serialize()); }

    p->SetValue("PeriodicDimensions", m_pimpl_->m_periodic_dimension_);
    p->SetValue("Dimensions", m_pimpl_->m_dimensions_);
    p->SetValue("IndexOrigin", m_pimpl_->m_idx_origin_);
    return p;
}
void MeshBase::Deserialize(const std::shared_ptr<DataTable>& cfg) {
    m_pimpl_->m_chart_ = cfg->has("Coordinates") ? geometry::Chart::Create(cfg->Get("Coordinates"))
                                                 : geometry::Chart::Create("Cartesian");
    m_pimpl_->m_idx_origin_ = cfg->GetValue<nTuple<int, 3>>("IndexOrigin", nTuple<int, 3>{0, 0, 0});
    m_pimpl_->m_dimensions_ = cfg->GetValue<nTuple<int, 3>>("Dimensions", nTuple<int, 3>{1, 1, 1});
    m_pimpl_->m_periodic_dimension_ = cfg->GetValue<nTuple<int, 3>>("PeriodicDimension", nTuple<int, 3>{0, 0, 0});
}
index_tuple MeshBase::GetGhostWidth(int tag) const {
    auto blk = GetBlock();
    return blk == nullptr ? index_tuple{0, 0, 0} : blk->GetGhostWidth();
}

box_type MeshBase::GetBox() const {
    box_type res;
    index_tuple lo, hi;
    std::tie(lo, hi) = GetIndexBox(VERTEX);
    std::get<0>(res) = point(EntityId{
        .w = 0, .x = static_cast<int16_t>(lo[0]), .y = static_cast<int16_t>(lo[1]), .z = static_cast<int16_t>(lo[2])});
    std::get<1>(res) = point(EntityId{
        .w = 0, .x = static_cast<int16_t>(hi[0]), .y = static_cast<int16_t>(hi[1]), .z = static_cast<int16_t>(hi[2])});
    return res;
}
point_type MeshBase::global_coordinates(EntityId s, point_type const& pr) const {
    return m_pimpl_->m_chart_->map(local_coordinates(s, pr));
}
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

EntityRange MeshBase::GetRange(std::string const& k) const {
    ASSERT(!isModified());
    ASSERT(m_pimpl_->m_patch_ != nullptr);
    auto it = m_pimpl_->m_patch_->m_ranges.find(k);
    return (it != m_pimpl_->m_patch_->m_ranges.end()) ? it->second
                                                      : EntityRange{std::make_shared<EmptyRangeBase<EntityId>>()};
};

EntityRange MeshBase::GetBodyRange(int IFORM, std::string const& k) const {
    return GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + "_BODY");
};
EntityRange MeshBase::GetBoundaryRange(int IFORM, std::string const& k, bool is_parallel) const {
    auto res =
        (IFORM == VERTEX || IFORM == VOLUME)
            ? GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + "_BOUNDARY")
            : GetRange(k + "." + std::string(EntityIFORMName[IFORM]) + (is_parallel ? "_PARA" : "_PERP") + "_BOUNDARY");
    return res;
};
EntityRange MeshBase::GetParallelBoundaryRange(int IFORM, std::string const& k) const {
    return GetBoundaryRange(IFORM, k, true);
}
EntityRange MeshBase::GetPerpendicularBoundaryRange(int IFORM, std::string const& k) const {
    return GetBoundaryRange(IFORM, k, false);
}
EntityRange MeshBase::GetGhostRange(int IFORM) const {
    return GetRange("." + std::string(EntityIFORMName[IFORM]) + "_GHOST");
}
void MeshBase::Push(Patch* p) {
    Click();
    m_pimpl_->m_patch_ = p;
    SetBlock(m_pimpl_->m_patch_->GetBlock());
    AttributeGroup::Push(p);
    DoUpdate();
}
void MeshBase::Pop(Patch* p) {
    p->SetBlock(GetMesh()->GetBlock());
    AttributeGroup::Pop(p);
    m_pimpl_->m_patch_ = nullptr;
    Click();
    DoTearDown();
}

std::map<std::string, EntityRange>& MeshBase::GetRangeDict() { return m_pimpl_->m_patch_->m_ranges; };
std::map<std::string, EntityRange> const& MeshBase::GetRangeDict() const { return m_pimpl_->m_patch_->m_ranges; };
}  // {namespace engine
}  // namespace simpla
