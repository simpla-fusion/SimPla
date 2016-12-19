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

Attribute::Attribute(std::shared_ptr<DataBlock> const &d, std::shared_ptr<AttributeDesc> const &desc)
        : m_desc_(desc), m_data_(d) {};

Attribute::Attribute(AttributeCollection *c, std::shared_ptr<AttributeDesc> const &desc)
        : m_desc_(desc) { connect(c); };

Attribute::~Attribute() { disconnect(); }

void Attribute::accept(Patch *p) { accept(p->data(m_desc_->id())); }

void Attribute::accept(std::shared_ptr<DataBlock> const &d)
{
    post_process();
    m_data_ = d;
}


void Attribute::pre_process()
{
    if (is_valid()) { return; } else { concept::LifeControllable::pre_process(); }

    if (m_data_ != nullptr) { return; }
    else
    {
//        m_data_ = create_data_block(m_mesh_, nullptr);
//        m_data_->pre_process();
    }
    ASSERT(m_data_ != nullptr);
}

void Attribute::post_process()
{
    m_data_.reset();
//    m_mesh_.reset();
    if (!is_valid()) { return; } else { concept::LifeControllable::post_process(); }
}


void Attribute::clear()
{
    pre_process();
    m_data_->clear();
}


std::ostream &
AttributeDict::print(std::ostream &os, int indent) const
{
    for (auto const &item:m_map_)
    {
        os << std::setw(indent + 1) << " " << item.second->name() << " = {" << item.second << "}," << std::endl;
    }
    return os;
};

std::pair<std::shared_ptr<AttributeDesc>, bool>
AttributeDict::register_attr(std::shared_ptr<AttributeDesc> const &desc)
{
    auto res = m_map_.emplace(desc->id(), desc);
    return std::make_pair(res.first->second, res.second);
}

void AttributeDict::erase(id_type const &id)
{
    auto it = m_map_.find(id);
    if (it != m_map_.end())
    {
        m_key_id_.erase(it->second->name());
        m_map_.erase(it);
    }
}

void AttributeDict::erase(std::string const &id)
{
    auto it = m_key_id_.find(id);
    if (it != m_key_id_.end()) { erase(it->second); }
}

std::shared_ptr<AttributeDesc> AttributeDict::find(id_type const &id)
{
    auto it = m_map_.find(id);
    if (it != m_map_.end()) { return it->second; } else { return nullptr; }
};

std::shared_ptr<AttributeDesc> AttributeDict::find(std::string const &id)
{
    auto it = m_key_id_.find(id);
    if (it != m_key_id_.end()) { return find(it->second); } else { return nullptr; }
};

std::shared_ptr<AttributeDesc> const &
AttributeDict::get(std::string const &k) const
{
    return m_map_.at(m_key_id_.at(k));
}

std::shared_ptr<AttributeDesc> const &
AttributeDict::get(id_type k) const
{
    return m_map_.at(k);
}

AttributeCollection::AttributeCollection(std::shared_ptr<AttributeDict> const &dict)
        : m_dict_(dict)
{
};

AttributeCollection::~AttributeCollection()
{

};

void
AttributeCollection::connect(Attribute *attr)
{
    base_type::connect(attr);

};

void
AttributeCollection::disconnect(Attribute *attr)
{
    base_type::disconnect(attr);
};
}}//namespace simpla { namespace mesh
