//
// Created by salmon on 16-10-20.
//

#include <typeindex>
#include <simpla/toolbox/Log.h>
#include "simpla/manifold/Atlas.h"
#include "Attribute.h"
#include "MeshBlock.h"
#include "DataBlock.h"
#include <simpla/manifold/Patch.h>

namespace simpla { namespace mesh
{


Attribute::Attribute(AttributeCollection *c) { connect(c); };

Attribute::~Attribute() { disconnect(); }


void Attribute::notify(Patch &p)
{
    m_mesh_block_ = p.mesh();
    m_data_ = p.data(name());
}

DataBlock *Attribute::data_block() { return m_data_.get(); };

DataBlock const *Attribute::data_block() const { return m_data_.get(); };

void Attribute::move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d)
{
    if (m == nullptr || m == m_mesh_block_) { return; }
    post_process();
    m_mesh_block_ = m;
    m_data_ = d;
}


void Attribute::pre_process()
{
    if (is_valid()) { return; } else { concept::LifeControllable::pre_process(); }

    ASSERT(m_mesh_block_ != nullptr);
    if (m_data_ != nullptr) { return; }
    else
    {
        m_data_ = create_data_block(m_mesh_block_, nullptr);
        m_data_->pre_process();
    }
    ASSERT(m_data_ != nullptr);
}

void Attribute::post_process()
{
    if (!is_valid()) { return; } else { concept::LifeControllable::post_process(); }
    m_data_.reset();
    m_mesh_block_.reset();
}


void Attribute::clear()
{
    pre_process();
    m_data_->clear();
}


}}//namespace simpla { namespace mesh
