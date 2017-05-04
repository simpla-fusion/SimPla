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
    struct range_group {
        Range<EntityId> body[4];
        Range<EntityId> boundary[4];
    };

    std::map<std::string, std::shared_ptr<geometry::GeoObject>> m_geo_object_;
    std::map<std::string, std::shared_ptr<range_group>> m_range_;
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

Range<EntityId> const* Domain::GetBody(std::string const& k) const {
    auto it = m_pimpl_->m_range_.find(k);
    return (it == m_pimpl_->m_range_.end() || it->second == nullptr) ? nullptr : it->second->body;
};
Range<EntityId> const* Domain::GetBoundary(std::string const& k) const {
    auto it = m_pimpl_->m_range_.find(k);
    return (it == m_pimpl_->m_range_.end() || it->second == nullptr) ? nullptr : it->second->boundary;
}

void Domain::Push(Patch* p) {
    GetMesh()->SetBlock(p->GetBlock());
    AttributeGroup::Push(p);
}
void Domain::Pop(Patch* p) {
    p->SetBlock(GetMesh()->GetBlock());
    AttributeGroup::Pop(p);
}

void Domain::InitialCondition(Real time_now) {
    if (GetMesh() == nullptr) { return; }

    GetMesh()->InitializeData(time_now);
    for (auto const& item : m_pimpl_->m_geo_object_) {
        CHECK(item.first);
        auto it = m_pimpl_->m_range_.emplace(item.first, std::make_shared<pimpl_s::range_group>());
        GetMesh()->InitializeRange(item.second, it.first->second->body, it.first->second->boundary);
    }
    OnInitialCondition(this, time_now);
}
void Domain::BoundaryCondition(Real time_now, Real dt) { OnBoundaryCondition(this, time_now, dt); }
void Domain::Advance(Real time_now, Real dt) { OnAdvance(this, time_now, dt); }

}  // namespace engine{

}  // namespace simpla{