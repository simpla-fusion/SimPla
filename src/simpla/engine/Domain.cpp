//
// Created by salmon on 17-4-5.
//

#include "Domain.h"
#include "Attribute.h"
#include "MeshBase.h"
#include "Patch.h"

namespace simpla {
namespace engine {

Domain::Domain(std::shared_ptr<geometry::GeoObject> g, std::shared_ptr<MeshBase> m) : m_mesh_(m), m_geo_object_(g) {}
Domain::~Domain() {}

std::shared_ptr<data::DataTable> Domain::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetClassName());
    return p;
}
void Domain::Deserialize(const std::shared_ptr<DataTable> &t) { UNIMPLEMENTED; };

std::shared_ptr<Domain> Domain::Clone() const {
    UNIMPLEMENTED;
    return nullptr;
}

void Domain::SetUp() { GetMesh()->SetUp(); }
void Domain::TearDown() { GetMesh()->TearDown(); }
void Domain::Initialize() { GetMesh()->Initialize(); }
void Domain::Finalize() { GetMesh()->Finalize(); }

void Domain::Push(Patch* p) { GetMesh()->Push(p); }
void Domain::Pop(Patch* p) { return GetMesh()->Pop(p); }

void Domain::InitializeCondition(Real time_now) { GetMesh()->InitializeData(time_now); }
void Domain::BoundaryCondition(Real time_now, Real dt) {}
void Domain::Advance(Real time_now, Real dt) {}

}  // namespace engine{

}  // namespace simpla{