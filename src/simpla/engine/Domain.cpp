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
    std::shared_ptr<std::map<std::string, nTuple<EntityRange, 4>>> m_range_;
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

EntityRange const* Domain::GetRange(std::string const& k) const {
    EntityRange const* res = nullptr;
    if (m_pimpl_->m_range_ != nullptr) {
        auto it = m_pimpl_->m_range_->find(k);
        res = (it == m_pimpl_->m_range_->end()) ? nullptr : &(it->second[0]);
    }
    return res;
};

EntityRange const* Domain::GetBody(std::string const& k) const { return GetRange(k); }
EntityRange const* Domain::GetBoundary(std::string const& k) const { return GetRange(k + ".boundary"); }

void Domain::Push(Patch* p) {
    GetMesh()->SetBlock(p->GetBlock());
    m_pimpl_->m_range_ = p->PopRange();
    AttributeGroup::Push(p, GetBody());
}
void Domain::Pop(Patch* p) {
    p->SetBlock(GetMesh()->GetBlock());
    p->PushRange(m_pimpl_->m_range_);
    AttributeGroup::Pop(p);
}

void Domain::InitialCondition(Real time_now) {
    if (GetMesh() == nullptr) { return; }

    GetMesh()->InitializeData(time_now);
    m_pimpl_->m_range_ = std::make_shared<std::map<std::string, nTuple<EntityRange, 4>>>();
    for (auto const& item : m_pimpl_->m_geo_object_) {
        auto body_it = m_pimpl_->m_range_->emplace(item.first, nTuple<EntityRange, 4>());
        auto boundary_it = m_pimpl_->m_range_->emplace(item.first + ".boundary", nTuple<EntityRange, 4>());
        GetMesh()->InitializeRange(item.second, &(body_it.first->second[0]), &(boundary_it.first->second[0]));
    }
    OnInitialCondition(this, time_now);
}
void Domain::BoundaryCondition(Real time_now, Real dt) { OnBoundaryCondition(this, time_now, dt); }
void Domain::Advance(Real time_now, Real dt) { OnAdvance(this, time_now, dt); }

}  // namespace engine{

}  // namespace simpla{