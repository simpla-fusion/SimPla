//
// Created by salmon on 16-10-20.
//

#include <typeindex>
#include <simpla/toolbox/Log.h>
#include "Atlas.h"
#include "Attribute.h"
#include "MeshBlock.h"
#include "DataBlock.h"
#include <simpla/mesh/Patch.h>

namespace simpla { namespace mesh
{

Attribute::Attribute() {};

Attribute::Attribute(AttributeCollection *c) { connect(c); };

Attribute::~Attribute() { disconnect(); }

void Attribute::accept(Patch *p) { accept(p->mesh(), p->data(m_desc_->id())); }

void Attribute::accept(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d)
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
