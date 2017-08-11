//
// Created by salmon on 17-7-16.
//

#include <simpla/geometry/BoxUtilities.h>
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

struct pack_s : public Patch::DataPack_s {
    std::map<std::string, Range<EntityId>> m_ranges_;
};

struct MeshBase::pimpl_s {
    std::shared_ptr<pack_s> m_pack_;
};

MeshBase::MeshBase() : m_pimpl_(new pimpl_s) {}

MeshBase::~MeshBase() = default;

int MeshBase::GetNDIMS() const { return 3; }

std::tuple<Real, index_box_type> MeshBase::CheckOverlap(const geometry::GeoObject* g) const {
    Real ratio = 0;
    if (g != nullptr) {
        auto b = GetBox(0);
        ratio = geometry::Measure(geometry::Overlap(g->BoundingBox(), b)) / geometry::Measure(b);
    }
    return std::make_tuple(ratio, IndexBox(0b0));
}

index_box_type MeshBase::IndexBox(int tag) const { return GetBlock()->IndexBox(); }

box_type MeshBase::GetBox(int tag) const {
    auto id_box = IndexBox(tag);
    return box_type{GetChart()->local_coordinates(std::get<0>(id_box), tag),
                    GetChart()->local_coordinates(std::get<1>(id_box), tag)};
};

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

std::shared_ptr<data::DataTable> MeshBase::Serialize() const { return base_type::Serialize(); }
void MeshBase::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    base_type::Deserialize(cfg);

    auto lo = cfg->GetValue<point_type>("Box/lo", point_type{0, 0, 0});
    auto hi = cfg->GetValue<point_type>("Box/hi", point_type{1, 1, 1});

    nTuple<int, 3> dims = cfg->GetValue("Dimensions", nTuple<int, 3>{1, 1, 1});

    GetChart()->SetOrigin(lo);
    GetChart()->SetScale((hi - lo) / (dims + 1));

    GetChart()->Deserialize(cfg->GetTable("Chart"));
    Click();
};

void MeshBase::DoUpdate() {
    SPObject::DoUpdate();
    if (m_pimpl_->m_pack_ == nullptr) { m_pimpl_->m_pack_ = std::make_shared<pack_s>(); }
    GetChart()->SetLevel(m_mesh_block_.GetLevel());
    AttributeGroup::RegisterAttributes();
}
void MeshBase::DoTearDown() { GetChart()->SetLevel(0); }
void MeshBase::DoInitialize() { GetChart()->SetLevel(m_mesh_block_.GetLevel()); }
void MeshBase::DoFinalize() {
    m_pimpl_->m_pack_.reset();
    GetChart()->SetLevel(0);
}

void MeshBase::SetBlock(const engine::MeshBlock& blk) { MeshBlock(blk).swap(m_mesh_block_); };
const MeshBlock* MeshBase::GetBlock() const { return &m_mesh_block_; }

void MeshBase::Push(Patch* patch) {
    //    VERBOSE << " Patch Level:" << patch->GetMeshBlock()->GetLevel() << " ID: " <<
    //    patch->GetMeshBlock()->GetLocalID()
    //            << " Block:" << patch->GetMeshBlock()->IndexBox() << std::endl;

    SetBlock(*patch->GetMeshBlock());
    GetChart()->SetLevel(GetBlock()->GetLevel());
    AttributeGroup::Push(patch);
    if (m_pimpl_->m_pack_ == nullptr) { m_pimpl_->m_pack_ = std::dynamic_pointer_cast<pack_s>(patch->GetDataPack()); }
    Update();
    ASSERT(GetBlock()->GetLevel() == GetChart()->GetLevel());
}
void MeshBase::Pop(Patch *patch) {
    patch->SetMeshBlock(*GetBlock());
    AttributeGroup::Pop(patch);
    patch->SetDataPack(m_pimpl_->m_pack_);

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
    if (m_pimpl_->m_pack_ == nullptr) { return Range<EntityId>{}; };
    auto it = m_pimpl_->m_pack_->m_ranges_.find(k);
    return (it == m_pimpl_->m_pack_->m_ranges_.end()) ? Range<EntityId>{} : it->second;
};

void MeshBase::InitialCondition(Real time_now) { DoInitialCondition(time_now); }
void MeshBase::BoundaryCondition(Real time_now, Real dt) { DoBoundaryCondition(time_now, dt); }
void MeshBase::Advance(Real time_now, Real dt) { DoAdvance(time_now, dt); }
void MeshBase::TagRefinementCells(Real time_now) { DoTagRefinementCells(time_now); }

void MeshBase::InitialCondition(Patch* patch, Real time_now) {
    Push(patch);
    InitialCondition(time_now);
    Pop(patch);
}
void MeshBase::BoundaryCondition(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    BoundaryCondition(time_now, dt);
    Pop(patch);
}
void MeshBase::Advance(Patch* patch, Real time_now, Real dt) {
    Push(patch);
    Advance(time_now, dt);
    Pop(patch);
}

}  // namespace engine{
}  // namespace simpla{