//
// Created by salmon on 16-11-4.
//
#include "Worker.h"
#include <set>
#include <simpla/mesh/MeshBlock.h>

namespace simpla { namespace simulation
{
struct Worker::pimpl_s
{
    std::set<Observer *> m_observers_;
};

Worker::Worker() : m_pimpl_(new pimpl_s) {}

Worker::~Worker() { for (Observer *ob:m_pimpl_->m_observers_) { detach(ob); }};

void Worker::attach(Observer *ob) { m_pimpl_->m_observers_.insert(ob); }

void Worker::detach(Observer *ob)
{
    ob->destroy();
    m_pimpl_->m_observers_.erase(ob);
};

void Worker::destroy() const { for (auto &ob:m_pimpl_->m_observers_) { ob->destroy(); }}

void Worker::create(mesh::MeshBlock const *m) const { for (auto &ob:m_pimpl_->m_observers_) { ob->create(m); }}

void Worker::move_to(mesh::MeshBlock const *m) const { for (auto &ob:m_pimpl_->m_observers_) { ob->move_to(m); }}

void Worker::deploy(mesh::MeshBlock const *m) const { for (auto &ob:m_pimpl_->m_observers_) { ob->deploy(m); }}

void Worker::erase(mesh::MeshBlock const *m) const { for (auto &ob:m_pimpl_->m_observers_) { ob->erase(m); }}

void Worker::update(mesh::MeshBlock const *m, bool og) const
{
    for (auto &ob:m_pimpl_->m_observers_)
    {
        ob->update(m, og);
    }
};


Worker::Observer::Observer(Worker *w) : m_worker_(w) { m_worker_->attach(this); }

Worker::Observer::~Observer() { m_worker_->detach(this); }
}}//namespace simpla { namespace mesh
