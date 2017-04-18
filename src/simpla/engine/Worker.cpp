//
// Created by salmon on 17-4-5.
//

#include "Worker.h"
#include "Attribute.h"
#include "Mesh.h"
#include "Patch.h"

namespace simpla {
namespace engine {

Worker::Worker() {}
Worker::~Worker() {}

std::shared_ptr<data::DataTable> Worker::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetClassName());
    return p;
}
void Worker::Deserialize(std::shared_ptr<data::DataTable>){};

Mesh *Worker::GetMesh() { return nullptr; }
Mesh const *Worker::GetMesh() const { return nullptr; }

void Worker::Register(AttributeGroup *attr_grp) { GetMesh()->Register(attr_grp); }
void Worker::Deregister(AttributeGroup *attr_grp) { GetMesh()->Deregister(attr_grp); }

void Worker::Push(Patch *p) { GetMesh()->Push(p); }
void Worker::Pop(Patch *p) { GetMesh()->Pop(p); }

void Worker::SetUp() { GetMesh()->SetUp(); }
void Worker::TearDown() { GetMesh()->TearDown(); }
void Worker::Initialize() { GetMesh()->Initialize(); }
void Worker::Finalize() { GetMesh()->Finalize(); }

void Worker::InitializeData(Real time_now) { GetMesh()->InitializeData(time_now); }
void Worker::AdvanceData(Real time_now, Real dt) {}

}  // namespace engine{

}  // namespace simpla{