//
// Created by salmon on 16-10-20.
//

#include "Atlas.h"
#include "Attribute.h"

namespace simpla { namespace mesh
{
struct Attribute::pimpl_s
{
    std::map<id_type, std::shared_ptr<DataBlock>> m_patches_;
    id_type m_id_;
};

Attribute::Attribute(std::string const &s) : toolbox::Object(s), m_pimpl_(new pimpl_s) {}

Attribute::~Attribute() {}

std::ostream &Attribute::print(std::ostream &os, int indent) const
{
    os << std::setw(indent) << " " << std::endl;
    for (auto const &item:m_pimpl_->m_patches_)
    {
        os << std::setw(indent) << " " << item.first << " = {";
        item.second->print(os, indent + 1);
        os << " } " << std::endl;
    }
    return os;
}


void Attribute::load(const data::DataBase &) { UNIMPLEMENTED; }

void Attribute::save(data::DataBase *) const { UNIMPLEMENTED; }

bool Attribute::has(const MeshBlock *m) const
{
    return m_pimpl_->m_patches_.find(m->id()) != m_pimpl_->m_patches_.end();
}

void Attribute::deploy(const MeshBlock *m) { m_pimpl_->m_patches_.at(m->id())->deploy(); }

void Attribute::erase(const MeshBlock *m) { m_pimpl_->m_patches_.erase(m->id()); }

void Attribute::clear(const MeshBlock *m) { m_pimpl_->m_patches_.at(m->id())->clear(); }

void Attribute::update(const MeshBlock *dest, const MeshBlock *src) { UNIMPLEMENTED; }


DataBlock const *Attribute::at(const MeshBlock *m) const
{
    return m_pimpl_->m_patches_.at(m == nullptr ? m_pimpl_->m_id_ : m->id()).get();
}

DataBlock *Attribute::at(const MeshBlock *m, const MeshBlock *hint)
{
    auto it = m_pimpl_->m_patches_.find(m->id());
    if (m_pimpl_->m_patches_.end() != it) { return it->second.get(); }
    else if (hint != nullptr)
    {
        try
        {
            auto res = at(hint)->create(m);
            insert(m, res);
            return res.get();
        } catch (std::out_of_range const &err) {}
    }
    return nullptr;

}

void Attribute::insert(const MeshBlock *m, const std::shared_ptr<DataBlock> &d)
{
    m_pimpl_->m_patches_.emplace(std::make_pair(m->id(), d));
}


}}//namespace simpla { namespace mesh
