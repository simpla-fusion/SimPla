//
// Created by salmon on 16-6-6.
//
#include "MeshAttribute.h"
#include "../parallel/DistributedObject.h"

namespace simpla { namespace mesh
{
struct MeshAttribute::View::pimpl_s
{
    parallel::DistributedObject m_dist_obj_;
};

MeshAttribute::View::View() : m_pimpl_(new pimpl_s) { }

MeshAttribute::View::~View() { }

void MeshAttribute::View::sync(bool is_blocking)
{
    m_pimpl_->m_dist_obj_.sync();
    if (is_blocking) { wait(); }
}

void MeshAttribute::View::wait()
{
    LOG_CMD_DESC(" SYNC [" + get_class_name() + "]", m_pimpl_->m_dist_obj_.wait());
}

bool MeshAttribute::View::is_ready() const
{
    return m_pimpl_->m_dist_obj_.is_ready();
}
}}//namespace simpla{namespace get_mesh{