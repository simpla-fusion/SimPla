//
// Created by salmon on 17-4-5.
//

#include "Worker.h"
#include "Attribute.h"
#include "Mesh.h"
#include "Patch.h"
#include "Task.h"

namespace simpla {
namespace engine {
struct Worker::pimpl_s {
    std::shared_ptr<Mesh> m_mesh_ = nullptr;
};
Worker::Worker() : m_pimpl_(new pimpl_s), concept::Configurable() {}
Worker::Worker(Worker const &other) : m_pimpl_(new pimpl_s), concept::Configurable(other) {
    m_pimpl_->m_mesh_ = other.m_pimpl_->m_mesh_;
}
Worker::~Worker() {}
void Worker::swap(Worker &other) {
    concept::Configurable::swap(other);
    std::swap(m_pimpl_->m_mesh_, other.m_pimpl_->m_mesh_);
}
Worker *Worker::Clone() const { return new Worker(*this); };
void Worker::Register(AttributeBundle *) {}
void Worker::Deregister(AttributeBundle *) {}
void Worker::SetMesh(std::shared_ptr<Mesh> const &m) { m_pimpl_->m_mesh_ = m; }
std::shared_ptr<Mesh> const &Worker::GetMesh() const { return m_pimpl_->m_mesh_; }
void Worker::Initialize(Real time_now) {}
void Worker::Advance(Real time_now, Real dt) { Initialize(time_now); }
void Worker::Finalize() {}

}  // namespace engine{

}  // namespace simpla{