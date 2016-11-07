//
// Created by salmon on 16-11-4.
//
#include "Worker.h"
#include <set>
#include <simpla/mesh/MeshBlock.h>

namespace simpla { namespace mesh
{
struct Worker::pimpl_s
{
    MeshBlock const *m_mesh_ = nullptr;
    std::set<Observer *> m_obs_;
};

Worker::Worker() : m_pimpl_(new pimpl_s) {}

Worker::~Worker() { for (Observer *ob:m_pimpl_->m_obs_) { ob->destroy(); }};

std::ostream &Worker::print(std::ostream &os, int indent) const
{
    os << std::setw(indent + 1) << " Mesh = " << m_pimpl_->m_mesh_->name() << ", "
       << " type = \"" << get_class_name() << "\", "
       << "Attribute= {";

    for (Observer *ob:m_pimpl_->m_obs_) { os << "\"" << ob->name() << "\" , "; }

    os << std::setw(indent + 1) << "}  ";

    return os;
}

void Worker::attach(Observer *ob) { m_pimpl_->m_obs_.insert(ob); }

void Worker::detach(Observer *ob) { if (ob != nullptr) { m_pimpl_->m_obs_.erase(ob); }};

void Worker::apply(Visitor const &vis) { for (auto &ob:m_pimpl_->m_obs_) { vis.visit(*ob); }}

void Worker::apply(Visitor const &vis) const { for (auto &ob:m_pimpl_->m_obs_) { vis.visit(*ob); }}

void Worker::apply(std::function<void(Observer &)> const &f) { for (auto &ob:m_pimpl_->m_obs_) { f(*ob); }}

void Worker::apply(std::function<void(Observer const &)> const &f) const { for (auto &ob:m_pimpl_->m_obs_) { f(*ob); }}

void Worker::move_to(mesh::MeshBlock const *m)
{
    m_pimpl_->m_mesh_ = m;
    for (auto &ob:m_pimpl_->m_obs_) { ob->move_to(m); }
}

void Worker::deploy()
{
    move_to(m_pimpl_->m_mesh_);
    for (auto &ob:m_pimpl_->m_obs_) { ob->deploy(); }
}

void Worker::destroy()
{
    for (auto &ob:m_pimpl_->m_obs_) { ob->destroy(); }
    m_pimpl_->m_mesh_ = nullptr;
}

void Worker::sync(MeshBlock const *other, bool og) { for (auto *ob:m_pimpl_->m_obs_) { ob->sync(other, og); }};

Worker::Observer::Observer(Worker *w) : m_worker_(w) { if (m_worker_ != nullptr) { m_worker_->attach(this); }}

Worker::Observer::~Observer() { if (m_worker_ != nullptr) { m_worker_->detach(this); }}

}}//namespace simpla { namespace mesh1
