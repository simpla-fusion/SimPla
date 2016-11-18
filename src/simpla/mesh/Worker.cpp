//
// Created by salmon on 16-11-4.
//
#include "Worker.h"
#include <set>
#include "MeshBlock.h"
#include "Attribute.h"

namespace simpla { namespace mesh
{
struct Worker::pimpl_s
{
    MeshBlock const *m_mesh_ = nullptr;
    std::vector<std::shared_ptr<Attribute>> m_attributes_;
    std::shared_ptr<Atlas> m_atlas_;
};

Worker::Worker() : m_pimpl_(new pimpl_s) {}

Worker::~Worker() {};

std::ostream &Worker::print(std::ostream &os, int indent) const
{
    if (m_pimpl_->m_mesh_ != nullptr)
    {
        os << std::setw(indent + 1) << " Mesh = " << m_pimpl_->m_mesh_->name() << ", "
           << " type = \"" << get_class_name() << "\", ";

    }
    os << "Attribute= {";

//    foreach([&](AttributeViewBase const &ob) { os << "\"" << ob.attribute()->name() << "\" , "; });

    os << std::setw(indent + 1) << "}  ";

    return os;
}


void Worker::move_to(const MeshBlock *m)
{
    m_pimpl_->m_mesh_ = m;
//    notify(m);
}

MeshBlock const *Worker::mesh_block() const { return m_pimpl_->m_mesh_; }

void Worker::deploy()
{
    move_to(m_pimpl_->m_mesh_);
//    foreach([&](AttributeViewBase &ob) { ob.deploy(); });

}

void Worker::destroy()
{
//    foreach([&](AttributeViewBase &ob) { ob.destroy(); });
    m_pimpl_->m_mesh_ = nullptr;
}


}}//namespace simpla { namespace mesh1
