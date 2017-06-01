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
#include "Model.h"
#include "Patch.h"
namespace simpla {
namespace engine {

struct MeshBase::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_block_;
    std::shared_ptr<geometry::Chart> m_chart_;
    std::shared_ptr<Patch> m_patch_ = nullptr;
};
MeshBase::MeshBase(std::shared_ptr<geometry::Chart> const& c, std::string const& s_name)
    : SPObject(s_name), AttributeGroup(), m_pimpl_(new pimpl_s) {
    m_pimpl_->m_chart_ = c;
}
MeshBase::~MeshBase() {}
void MeshBase::GetChart(std::shared_ptr<geometry::Chart> const& c) {
    m_pimpl_->m_chart_ = c;
    Click();
}
std::shared_ptr<geometry::Chart> MeshBase::GetChart() const { return m_pimpl_->m_chart_; }

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
    return p;
}
void MeshBase::Deserialize(const std::shared_ptr<DataTable>&) {}
index_tuple MeshBase::GetGhostWidth(int tag) const {
    auto blk = GetBlock();
    return blk == nullptr ? index_tuple{0, 0, 0} : blk->GetGhostWidth();
}

point_type MeshBase::global_coordinates(EntityId s, point_type const& pr) const {
    return GetChart()->map(local_coordinates(s, pr));
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
    auto it = m_pimpl_->m_patch_->m_ranges.find(k);
    //    VERBOSE << FILE_LINE_STAMP << "Get Range " << k << std::endl;
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
void MeshBase::Push(const std::shared_ptr<Patch>& p) {
    Click();
    m_pimpl_->m_patch_ = p;
    SetBlock(m_pimpl_->m_patch_->GetBlock());
    for (auto& item : GetAllAttributes()) {
        auto k = "." + std::string(EntityIFORMName[item.second->GetIFORM()]) + "_BODY";

        auto it = m_pimpl_->m_patch_->m_ranges.find(k);
        item.second->Push(m_pimpl_->m_patch_->Pop(item.second->GetID()),
                          (it == m_pimpl_->m_patch_->m_ranges.end()) ? EntityRange{} : it->second);
    }

    DoUpdate();
}
std::shared_ptr<Patch> MeshBase::PopPatch() {
    m_pimpl_->m_patch_->SetBlock(GetMesh()->GetBlock());
    for (auto& item : GetMesh()->GetAllAttributes()) {
        m_pimpl_->m_patch_->Push(item.second->GetID(), item.second->Pop());
    }
    auto res = m_pimpl_->m_patch_;
    m_pimpl_->m_patch_ = nullptr;
    Click();
    DoTearDown();
    return res;
}

std::map<std::string, EntityRange>& MeshBase::GetRangeDict() { return m_pimpl_->m_patch_->m_ranges; };
std::map<std::string, EntityRange> const& MeshBase::GetRangeDict() const { return m_pimpl_->m_patch_->m_ranges; };
}  // {namespace engine
}  // namespace simpla
