//
// Created by salmon on 17-4-5.
//

#include "Worker.h"
#include "Attribute.h"
#include "Mesh.h"
#include "Patch.h"

namespace simpla {
namespace engine {

// struct WorkerFactory {
//    std::map<std::string, std::function<Worker *()>> m_task_factory_;
//};
// bool Worker::RegisterCreator(std::string const &k, std::function<Worker *()> const &fun) {
//    auto res = SingletonHolder<WorkerFactory>::instance().m_task_factory_.emplace(k, fun).second;
//
//    if (res) { LOGGER << "Work Creator [ " << k << " ] is registered!" << std::endl; }
//
//    return res;
//}
// Worker *Worker::Create(std::string const &s_name) {
//    //    std::string s_name = "";
//    //    std::shared_ptr<data::DataTable> d = nullptr;
//    //
//    //    if (config != nullptr) { s_name = m->name() + "." + config->GetValue<std::string>("name"); }
//    auto it = SingletonHolder<WorkerFactory>::instance().m_task_factory_.find(s_name);
//    if (it != SingletonHolder<WorkerFactory>::instance().m_task_factory_.end()) {
//        auto res = it->second();
//        //        res->db() = config;
//        //        res->SetMesh(m.get());
//        //        LOGGER << "Work [" << s_name << "] is created!" << std::endl;
//        return res;
//    } else {
//        RUNTIME_ERROR << "Work Creator[" << s_name << "] is missing!" << std::endl;
//        return nullptr;
//    }
//}

struct Worker::pimpl_s {
    std::shared_ptr<Mesh> m_mesh_ = nullptr;
};
Worker::Worker() : m_pimpl_(new pimpl_s) {}
Worker::Worker(Worker const &other) : m_pimpl_(new pimpl_s) { m_pimpl_->m_mesh_ = other.m_pimpl_->m_mesh_; }
Worker::~Worker() {}
void Worker::swap(Worker &other) { std::swap(m_pimpl_->m_mesh_, other.m_pimpl_->m_mesh_); }
Worker *Worker::Clone() const { return new Worker(*this); };

void Worker::SetMesh(std::shared_ptr<Mesh> const &m) { m_pimpl_->m_mesh_ = m; }
std::shared_ptr<Mesh> const &Worker::GetMesh() const { return m_pimpl_->m_mesh_; }

void Worker::Register(AttributeGroup *attr_grp) {
    AttributeGroup::Register(attr_grp);
    if (m_pimpl_->m_mesh_ != nullptr) m_pimpl_->m_mesh_->Register(attr_grp);
}
void Worker::Deregister(AttributeGroup *attr_grp) {
    AttributeGroup::Deregister(attr_grp);
    if (m_pimpl_->m_mesh_ != nullptr)   m_pimpl_->m_mesh_->Deregister(attr_grp);
}

void Worker::Push(Patch *p) {
    //    SetMesh(p->GetMeshBlock());
}
void Worker::Pop(Patch *) {
    //    auto p = std::make_shared<Patch>();
    //    p->SetMesh(GetMesh());
    //    return std::move(p);
}

void Worker::Initialize(Real time_now) {}
void Worker::Advance(Real time_now, Real dt) { Initialize(time_now); }
void Worker::Finalize() {}

}  // namespace engine{

}  // namespace simpla{