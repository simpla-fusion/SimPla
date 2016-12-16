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


void Attribute::notify(std::shared_ptr<Patch> const &p)
{
    post_process();
    move_to(p->mesh(), p->data(m_desc_->name()));
}

void Attribute::move_to(std::shared_ptr<Chart> const &m, std::shared_ptr<DataBlock> const &d)
{
    if (m == nullptr || m == m_mesh_) { return; }
    post_process();
    m_mesh_ = m;
    m_data_ = d;
}


void Attribute::pre_process()
{
    if (is_valid()) { return; } else { concept::LifeControllable::pre_process(); }

    ASSERT(m_mesh_ != nullptr);
    if (m_data_ != nullptr) { return; }
    else
    {
        m_data_ = create_data_block(m_mesh_, nullptr);
        m_data_->pre_process();
    }
    ASSERT(m_data_ != nullptr);
}

void Attribute::post_process()
{
    m_data_.reset();
    m_mesh_.reset();
    if (!is_valid()) { return; } else { concept::LifeControllable::post_process(); }
}


void Attribute::clear()
{
    pre_process();
    m_data_->clear();
}


}}//namespace simpla { namespace mesh
