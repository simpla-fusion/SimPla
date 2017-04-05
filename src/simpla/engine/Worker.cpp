//
// Created by salmon on 17-4-5.
//

#include "Worker.h"
#include "Attribute.h"
#include "Mesh.h"
#include "Patch.h"
namespace simpla {
namespace engine {
struct Worker::pimpl_s {
    std::shared_ptr<Mesh> m_mesh_ = nullptr;
    std::vector<Attribute *> m_attr_;
};
Worker::Worker(std::shared_ptr<data::DataTable> const &t, std::shared_ptr<Mesh> const &m)
    : m_pimpl_(new pimpl_s), concept::Configurable(t) {
    m_pimpl_->m_mesh_ = std::shared_ptr<Mesh>(m->Clone());
}
Worker::Worker(Worker const &other) : m_pimpl_(new pimpl_s), concept::Configurable(other) {
    m_pimpl_->m_mesh_ = std::shared_ptr<Mesh>(other.m_pimpl_->m_mesh_->Clone());
}
Worker::~Worker() {}
void Worker::swap(Worker &other) {
    concept::Configurable::swap(other);
    std::swap(m_pimpl_->m_mesh_, other.m_pimpl_->m_mesh_);
}
Worker *Worker::Clone() const { return new Worker(*this); };

std::vector<Attribute *> Worker::GetAttributes() const { return std::vector<Attribute *>(); }
std::shared_ptr<Mesh> const &Worker::GetMesh() const { return m_pimpl_->m_mesh_; }
void Worker::PushData(std::shared_ptr<Patch> p) {
    m_pimpl_->m_mesh_->PushData(p);
    for (auto *attr : m_pimpl_->m_attr_) { attr->PushData(p->PopData(attr->GetGUID())); }
}
std::shared_ptr<Patch> Worker::PopData() {
    auto p = std::make_shared<Patch>();
    p->PushMeshBlock(m_pimpl_->m_mesh_->GetBlock());
    for (auto *attr : m_pimpl_->m_attr_) { p->PushData(attr->GetGUID(), attr->PopData()); }
    return p;
}

void Worker::Initialize(Real time_now) {}
void Worker::Advance(Real time_now, Real dt) { Initialize(time_now); }
void Worker::Finalize() {}
}  // namespace engine{

}  // namespace simpla{