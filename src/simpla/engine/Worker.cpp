//
// Created by salmon on 17-4-5.
//

#include "Worker.h"
#include "Attribute.h"
#include "MeshBase.h"
#include "Patch.h"

namespace simpla {
namespace engine {

Worker::Worker(std::shared_ptr<MeshBase> m) { m_mesh_ = m; }
Worker::~Worker() {}

std::shared_ptr<data::DataTable> Worker::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetClassName());
    return p;
}
void Worker::Deserialize(std::shared_ptr<data::DataTable> t){};

void Worker::SetUp() { GetMesh()->SetUp(); }
void Worker::TearDown() { GetMesh()->TearDown(); }
void Worker::Initialize() { GetMesh()->Initialize(); }
void Worker::Finalize() { GetMesh()->Finalize(); }

void Worker::Push(std::shared_ptr<Patch> p) { GetMesh()->Push(p); }
std::shared_ptr<Patch> Worker::Pop() { return GetMesh()->Pop(); }

void Worker::InitializeCondition(Real time_now) { GetMesh()->InitializeData(time_now); }
void Worker::BoundaryCondition(Real time_now, Real dt) {}
void Worker::Advance(Real time_now, Real dt) {}

}  // namespace engine{

}  // namespace simpla{