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
};
MeshBase::MeshBase(std::shared_ptr<geometry::Chart> const &c) : SPObject(), AttributeGroup(), m_pimpl_(new pimpl_s) {
    m_pimpl_->m_chart_ = c;
}
MeshBase::~MeshBase() {}

std::shared_ptr<geometry::Chart> MeshBase::GetChart() const { return m_pimpl_->m_chart_; }

void MeshBase::SetBlock(std::shared_ptr<MeshBlock> m) {
    m_pimpl_->m_mesh_block_ = m;
    Click();
}
std::shared_ptr<MeshBlock> MeshBase::GetBlock() const { return m_pimpl_->m_mesh_block_; }

id_type MeshBase::GetBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->GetGUID();
}

void MeshBase::InitializeData(Real time_now) {
    DoUpdate();
    GetChart()->InitializeData(this, time_now);
}
void MeshBase::SetBoundaryCondition(Real time_now, Real time_dt) { DoUpdate(); }

std::shared_ptr<data::DataTable> MeshBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    return p;
}
void MeshBase::Deserialize(const std::shared_ptr<DataTable> &) {}
index_tuple MeshBase::GetGhostWidth(int tag) const {
    auto blk = GetBlock();
    return blk == nullptr ? index_tuple{0, 0, 0} : blk->GetGhostWidth();
}
index_box_type MeshBase::GetIndexBox(int tag) const {
    index_box_type res = GetBlock()->GetIndexBox();
    switch (tag) {
        case 0:
            std::get<1>(res) += 1;
            break;
        case 1:
            std::get<1>(res)[1] += 1;
            std::get<1>(res)[2] += 1;
            break;
        case 2:
            std::get<1>(res)[0] += 1;
            std::get<1>(res)[2] += 1;
            break;
        case 4:
            std::get<1>(res)[0] += 1;
            std::get<1>(res)[1] += 1;
            break;
        case 3:
            std::get<1>(res)[2] += 1;
            break;
        case 5:
            std::get<1>(res)[1] += 1;
            break;
        case 6:
            std::get<1>(res)[0] += 1;
            break;
        case 7:
        default:
            break;
    }
    return res;
}

}  // {namespace engine
}  // namespace simpla
