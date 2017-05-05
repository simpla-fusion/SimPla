//
// Created by salmon on 17-4-5.
//

#include "Domain.h"
#include "Attribute.h"
#include "MeshBase.h"
#include "Patch.h"

namespace simpla {
namespace engine {

struct Domain::pimpl_s {
    std::map<std::string, std::shared_ptr<geometry::GeoObject>> m_geo_object_;
    std::shared_ptr<std::map<std::string, EntityRange>> m_range_;
};
Domain::Domain(std::shared_ptr<geometry::GeoObject> const& g) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_geo_object_[""] = g;
}
Domain::~Domain() {}

std::shared_ptr<data::DataTable> Domain::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetRegisterName());
    return p;
}
void Domain::Deserialize(const std::shared_ptr<DataTable>& t) { UNIMPLEMENTED; };

void Domain::SetUp() {
    SPObject::SetUp();
    GetMesh()->SetUp();
}
void Domain::TearDown() {
    GetMesh()->TearDown();
    SPObject::TearDown();
}
void Domain::Initialize() {
    GetMesh()->Initialize();
    SPObject::Initialize();
}
void Domain::Finalize() {
    GetMesh()->Finalize();
    SPObject::Finalize();
}

void Domain::SetGeoObject(std::string const& k, std::shared_ptr<geometry::GeoObject> const& g) {
    m_pimpl_->m_geo_object_[k] = g;
}

std::shared_ptr<geometry::GeoObject> Domain::GetGeoObject(std::string const& k) const {
    auto it = m_pimpl_->m_geo_object_.find(k);
    return (it == m_pimpl_->m_geo_object_.end()) ? nullptr : it->second;
}

EntityRange Domain::GetRange(std::string const& k) const {
    if (m_pimpl_->m_range_ != nullptr) {
        auto it = m_pimpl_->m_range_->find(k);
        if (it != m_pimpl_->m_range_->end()) { return it->second; }
    }
    return EntityRange{};
};

EntityRange Domain::GetBodyRange(int IFORM, std::string const& k) const {
    return GetRange(k + "." + EntityIFORMName[IFORM] + "_BODY");
};
EntityRange Domain::GetBoundaryRange(int IFORM, std::string const& k, bool is_parallel) const {
    return is_parallel ? GetParallelBoundaryRange(IFORM, k) : GetPerpendicularBoundaryRange(IFORM, k);
};
EntityRange Domain::GetParallelBoundaryRange(int IFORM, std::string const& k) const {
    return (IFORM == VERTEX || IFORM == VOLUME) ? GetRange(k + "." + EntityIFORMName[IFORM] + "_BOUNDARY")
                                                : GetRange(k + "." + EntityIFORMName[IFORM] + "_PARA_BOUNDARY");
}
EntityRange Domain::GetPerpendicularBoundaryRange(int IFORM, std::string const& k) const {
    return (IFORM == VERTEX || IFORM == VOLUME) ? GetRange(k + "." + EntityIFORMName[IFORM] + "_BOUNDARY")
                                                : GetRange(k + "." + EntityIFORMName[IFORM] + "_PERP_BOUNDARY");
}

void Domain::Push(Patch* p) {
    GetMesh()->SetBlock(p->GetBlock());
    m_pimpl_->m_range_ = p->PopRange();

    for (auto& item : GetAll()) {
        item.second->Push(p->Pop(item.second->GetID()), GetBodyRange(item.second->GetIFORM()));
    }
}
void Domain::Pop(Patch* p) {
    p->SetBlock(GetMesh()->GetBlock());
    p->PushRange(m_pimpl_->m_range_);
    for (auto& item : GetAll()) { p->Push(item.second->GetID(), item.second->Pop()); }
}

void Domain::InitialCondition(Real time_now) {
    if (GetMesh() == nullptr) { return; }

    GetMesh()->InitializeData(time_now);

    m_pimpl_->m_range_ = std::make_shared<std::map<std::string, EntityRange>>();

    for (auto const& item : m_pimpl_->m_geo_object_) {
        if (item.second == nullptr) { continue; }

        EntityRange r[10];
        GetMesh()->InitializeRange(item.second, r);

        m_pimpl_->m_range_->emplace(item.first + ".VERTEX_BODY", r[MeshBase::VERTEX_BODY]);
        m_pimpl_->m_range_->emplace(item.first + ".EDGE_BODY", r[MeshBase::EDGE_BODY]);
        m_pimpl_->m_range_->emplace(item.first + ".FACE_BODY", r[MeshBase::FACE_BODY]);
        m_pimpl_->m_range_->emplace(item.first + ".VOLUME_BODY", r[MeshBase::VOLUME_BODY]);
        m_pimpl_->m_range_->emplace(item.first + ".VERTEX_BOUNDARY", r[MeshBase::VERTEX_BOUNDARY]);
        m_pimpl_->m_range_->emplace(item.first + ".EDGE_PARA_BOUNDARY", r[MeshBase::EDGE_PARA_BOUNDARY]);
        m_pimpl_->m_range_->emplace(item.first + ".FACE_PARA_BOUNDARY", r[MeshBase::FACE_PARA_BOUNDARY]);
        m_pimpl_->m_range_->emplace(item.first + ".EDGE_PERP_BOUNDARY", r[MeshBase::EDGE_PERP_BOUNDARY]);
        m_pimpl_->m_range_->emplace(item.first + ".FACE_PERP_BOUNDARY", r[MeshBase::FACE_PERP_BOUNDARY]);
        m_pimpl_->m_range_->emplace(item.first + ".VOLUME_BOUNDARY", r[MeshBase::VOLUME_BOUNDARY]);
    }
    OnInitialCondition(this, time_now);
}
void Domain::BoundaryCondition(Real time_now, Real dt) { OnBoundaryCondition(this, time_now, dt); }
void Domain::Advance(Real time_now, Real dt) { OnAdvance(this, time_now, dt); }

}  // namespace engine{

}  // namespace simpla{