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

struct pack_s : public Patch::DataPack_s {
    std::map<std::string, Range<EntityId>> m_ranges_;
};

struct MeshBase::pimpl_s {
    std::shared_ptr<pack_s> m_pack_;
};

MeshBase::MeshBase(std::shared_ptr<geometry::Chart> const& c, index_box_type const& b)
    : m_pimpl_(new pimpl_s), m_chart_(c), m_mesh_block_(b, 0, 0, 0) {}

MeshBase::~MeshBase() = default;

index_box_type MeshBase::GetIndexBox(int tag) const { return GetBlock()->GetIndexBox(); }

box_type MeshBase::GetBox(int tag) const {
    auto id_box = GetIndexBox(tag);
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

std::shared_ptr<data::DataTable> MeshBase::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("Chart", GetChart()->Serialize());

    return (p);
}
void MeshBase::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    SetName(cfg->GetValue("Name", GetRegisterName()));

    auto lo = cfg->GetValue<point_type>("Box/lo", point_type{0, 0, 0});
    auto hi = cfg->GetValue<point_type>("Box/hi", point_type{1, 1, 1});

    auto dims = cfg->GetValue<nTuple<int, 3>>("Dimensions", nTuple<int, 3>{1, 1, 1});

    if (cfg->isTable("Chart")) {
        m_chart_ = geometry::Chart::Create(cfg->Get("Chart/Type"));

    } else {
        m_chart_ = geometry::Chart::Create(cfg->Get("Chart"));
    }
    m_chart_->SetOrigin(lo);
    m_chart_->SetScale((hi - lo) / dims);

    if (cfg->isTable("Chart")) { m_chart_->Deserialize(cfg->GetTable("Chart")); }
    Click();
};

void MeshBase::DoUpdate() {
    SPObject::DoUpdate();
    if (m_pimpl_->m_pack_ == nullptr) { m_pimpl_->m_pack_ = std::make_shared<pack_s>(); }
    m_chart_->SetLevel(m_mesh_block_.GetLevel());
    AttributeGroup::RegisterAttributes();
}
void MeshBase::DoTearDown() { m_chart_->SetLevel(0); }
void MeshBase::DoInitialize() { m_chart_->SetLevel(m_mesh_block_.GetLevel()); }
void MeshBase::DoFinalize() {
    m_pimpl_->m_pack_.reset();
    m_chart_->SetLevel(0);
}

void MeshBase::SetBlock(const engine::MeshBlock& blk) { MeshBlock(blk).swap(m_mesh_block_); };
const MeshBlock* MeshBase::GetBlock() const { return &m_mesh_block_; }

void MeshBase::Push(Patch* patch) {
    VERBOSE << " Patch Level:" << patch->GetMeshBlock()->GetLevel() << " ID: " << patch->GetMeshBlock()->GetLocalID()
            << " Block:" << patch->GetMeshBlock()->GetIndexBox() << std::endl;

    SetBlock(*patch->GetMeshBlock());

    m_chart_->SetLevel(GetBlock()->GetLevel());

    AttributeGroup::Push(patch);
    if (m_pimpl_->m_pack_ == nullptr) { m_pimpl_->m_pack_ = std::dynamic_pointer_cast<pack_s>(patch->GetDataPack()); }
    Update();
    ASSERT(GetBlock()->GetLevel() == m_chart_->GetLevel());
}
void MeshBase::Pull(Patch* patch) {
    patch->SetMeshBlock(*GetBlock());
    AttributeGroup::Pull(patch);
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
void MeshBase::TagRefinementCells(Real time_now) {
    VERBOSE << "TagRefinementCells \t:" << GetName() << std::endl;
    DoTagRefinementCells(time_now);
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