//
// Created by salmon on 16-10-20.
//

#include "Atlas.h"
#include "Attribute.h"

namespace simpla
{
namespace mesh
{
struct AttributeBase::pimpl_s
{
    std::map<id_type, std::shared_ptr<DataBlockBase>> m_patches_;
    id_type m_id_;
};

AttributeBase::AttributeBase(std::string const &s) : toolbox::Object(s), m_pimpl_(new pimpl_s) {}

AttributeBase::~AttributeBase() {}

std::ostream &AttributeBase::print(std::ostream &os, int indent) const
{
    for (auto const &item:m_pimpl_->m_patches_) { item.second->print(os, indent + 1); }
    return os;
}


void AttributeBase::load(const data::DataBase &) { UNIMPLEMENTED; }

void AttributeBase::save(data::DataBase *) const { UNIMPLEMENTED; }

bool AttributeBase::has(const id_type &t_id) const
{
    return m_pimpl_->m_patches_.find(t_id) != m_pimpl_->m_patches_.end();
}


void AttributeBase::deploy(id_type id) { m_pimpl_->m_patches_.at(id)->deploy(); }

void AttributeBase::erase(id_type id) { m_pimpl_->m_patches_.erase(id); }

void AttributeBase::clear(id_type id) { m_pimpl_->m_patches_.at(id)->clear(); }

void AttributeBase::update(id_type dest, id_type src) { UNIMPLEMENTED; }

DataBlockBase &AttributeBase::data(id_type id) { return *m_pimpl_->m_patches_.at(id == 0 ? m_pimpl_->m_id_ : id); }

DataBlockBase const &AttributeBase::data(id_type id) const
{
    return *m_pimpl_->m_patches_.at(id == 0 ? m_pimpl_->m_id_ : id);
}

DataBlockBase &AttributeBase::create(const MeshBlock *m, id_type hint)
{
    auto res = data(hint).create(m);
    insert(m->id(), res);
    return *res;
}

void AttributeBase::insert(id_type id, const std::shared_ptr<DataBlockBase> &d)
{
    m_pimpl_->m_patches_.emplace(std::make_pair(id, d));

}


}
}