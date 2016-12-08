//
// Created by salmon on 16-10-20.
//

#include <typeindex>
#include <simpla/toolbox/Log.h>
#include "simpla/manifold/Atlas.h"
#include "Attribute.h"
#include "MeshBlock.h"
#include "DataBlock.h"

namespace simpla { namespace mesh
{

class ChartBase;


AttributeBase::AttributeBase(std::string const &config_str) : Object() { db.parse(config_str); }

AttributeBase::~AttributeBase() {}

std::ostream &AttributeBase::print(std::ostream &os, int indent) const { return os; }


void AttributeBase::load(const data::DataEntityTable &) { UNIMPLEMENTED; }

void AttributeBase::save(data::DataEntityTable *) const { UNIMPLEMENTED; }

AttributeCollection::AttributeCollection() {}

AttributeCollection::~AttributeCollection() {}

std::ostream &AttributeCollection::print(std::ostream &os, int indent) const { return os; }

void AttributeCollection::load(const data::DataEntityTable &) { UNIMPLEMENTED; }

void AttributeCollection::save(data::DataEntityTable *) const { UNIMPLEMENTED; }

const AttributeBase *AttributeCollection::find(const key_type &k) const
{
    auto it = m_map_.find(k);
    if (it != m_map_.end()) { return it->second.get(); } else { return nullptr; }
}

AttributeBase *AttributeCollection::find(const key_type &k)
{
    auto it = m_map_.find(k);
    if (it != m_map_.end()) { return it->second.get(); } else { return nullptr; }
}

std::shared_ptr<AttributeBase> const &AttributeCollection::at(const key_type &k) const { return m_map_.at(k); }

std::shared_ptr<AttributeBase> &AttributeCollection::at(const key_type &k) { return m_map_.at(k); }

void AttributeCollection::erase(const key_type &k) { m_map_.erase(k); }

std::pair<std::shared_ptr<AttributeBase>, bool>
AttributeCollection::emplace(const key_type &k, const std::shared_ptr<AttributeBase> &p)
{
    auto res = m_map_.emplace(k, p);
    return std::make_pair(res.first->second, res.second);
}


}}//namespace simpla { namespace mesh
