//
// Created by salmon on 17-7-16.
//

//
// Created by salmon on 17-7-16.
//

#include "simpla/SIMPLA_config.h"

#include "Mesh.h"

#include "simpla/geometry/Chart.h"
#include "simpla/geometry/GeoObject.h"

#include "Attribute.h"
#include "Domain.h"
#include "Patch.h"
#include "SPObject.h"

namespace simpla {
namespace engine {

struct pack_s : public PatchDataPack {
    std::map<std::string, Range<EntityId>> m_ranges_;
};

struct MeshBase::pimpl_s {
    std::shared_ptr<pack_s> m_pack_;
};

MeshBase::MeshBase() : m_pimpl_(new pimpl_s) {}

MeshBase::~MeshBase() = default;

// MeshBase::MeshBase(MeshBase const& other) : SPObject(other), m_pimpl_(new pimpl_s),
// m_mesh_block_(other.m_mesh_block_) {
//    m_pimpl_->m_pack_ = other.m_pimpl_->m_pack_;
//}
//
// MeshBase::MeshBase(MeshBase&& other) noexcept
//    : SPObject(std::move(other)), m_pimpl_(std::move(other.m_pimpl_)), m_mesh_block_(std::move(other.m_mesh_block_))
//    {}
//
// void MeshBase::swap(MeshBase& other) {
//    SPObject::swap(other);
//    std::swap(m_pimpl_, other.m_pimpl_);
//
//    m_mesh_block_.swap(other.m_mesh_block_);
//}

std::shared_ptr<data::DataTable> MeshBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    return (p);
}
void MeshBase::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    m_chart_ = geometry::Chart::Create(cfg->GetTable("Chart"));

    Click();
};

void MeshBase::DoUpdate() {
    if (m_pimpl_->m_pack_ == nullptr) { m_pimpl_->m_pack_ = std::make_shared<pack_s>(); }
}
void MeshBase::DoTearDown() {}
void MeshBase::DoInitialize() {}
void MeshBase::DoFinalize() { m_pimpl_->m_pack_.reset(); }

void MeshBase::SetBlock(const engine::MeshBlock& blk) { MeshBlock(blk).swap(m_mesh_block_); };
const engine::MeshBlock& MeshBase::GetBlock() const { return m_mesh_block_; }
id_type MeshBase::GetBlockId() const { return m_mesh_block_.GetGUID(); }

void MeshBase::Push(Patch* patch) {
    SetBlock(patch->GetMeshBlock());
    AttributeGroup::Push(patch);
    if (m_pimpl_->m_pack_ == nullptr) {
        m_pimpl_->m_pack_ = std::dynamic_pointer_cast<pack_s>(patch->GetPack(GetName()));
    }
    Initialize();
}
void MeshBase::Pull(Patch* patch) {
    patch->SetMeshBlock(GetBlock());
    AttributeGroup::Pull(patch);
    patch->SetPack(GetName(), std::dynamic_pointer_cast<PatchDataPack>(m_pimpl_->m_pack_));
    Finalize();
}
void MeshBase::SetRange(std::string const& k, Range<EntityId> const& r) {
    Update();
    m_pimpl_->m_pack_->m_ranges_[k] = r;
}

Range<EntityId>& MeshBase::GetRange(std::string const& k) {
    Update();
    return m_pimpl_->m_pack_->m_ranges_[k];
};
Range<EntityId> MeshBase::GetRange(std::string const& k) const {
    ASSERT(m_pimpl_->m_pack_ != nullptr);
    auto it = m_pimpl_->m_pack_->m_ranges_.find(k);
    return (it == m_pimpl_->m_pack_->m_ranges_.end()) ? Range<EntityId>{} : it->second;
};

void MeshBase::InitialCondition(Real time_now) {
    VERBOSE << "InitialCondition   \t:" << GetName() << std::endl;
    DoInitialCondition(time_now);
}
void MeshBase::BoundaryCondition(Real time_now, Real dt) {
    VERBOSE << "Boundary Condition \t:" << GetName() << std::endl;
    DoBoundaryCondition(time_now, dt);
}
void MeshBase::Advance(Real time_now, Real dt) {
    VERBOSE << "Advance            \t:" << GetName() << std::endl;
    DoAdvance(time_now, dt);
}
void MeshBase::InitialCondition(Patch* patch, Real time_now) {
    Push(patch);
    InitialCondition(time_now);
    Pull(patch);
}
void MeshBase::BoundaryCondition(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    BoundaryCondition(time_now, dt);
    Pull(patch);
}
void MeshBase::Advance(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    Advance(time_now, dt);
    Pull(patch);
}

}  // namespace engine{
}  // namespace simpla{