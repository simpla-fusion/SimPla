//
// Created by salmon on 16-10-20.
//

#include <simpla/toolbox/Log.h>
#include "Atlas.h"
#include "Attribute.h"

namespace simpla { namespace mesh
{
struct Attribute::pimpl_s { std::map<id_type, std::shared_ptr<DataBlock>> m_patches_; };

Attribute::Attribute(std::string const &s) : toolbox::Object(s), m_pimpl_(new pimpl_s) {}

Attribute::~Attribute() {}

std::ostream &Attribute::print(std::ostream &os, int indent) const
{
    if (m_pimpl_ != nullptr)
    {
        os << std::setw(indent) << " " << std::endl;
        for (auto const &item:m_pimpl_->m_patches_)
        {
            os << std::setw(indent) << " " << item.second->mesh()->name() << " = {";
            item.second->print(os, indent + 1);
            os << " } " << std::endl;
        }
    }
    return os;
}


void Attribute::load(const data::DataBase &) { UNIMPLEMENTED; }

void Attribute::save(data::DataBase *) const { UNIMPLEMENTED; }

bool Attribute::has(const MeshBlock *m) const
{
    ASSERT(m_pimpl_ != nullptr);
    return (m == nullptr) ? false : m_pimpl_->m_patches_.find(m->id()) != m_pimpl_->m_patches_.end();
}


void Attribute::erase(const MeshBlock *m)
{
    ASSERT(m_pimpl_ != nullptr);
    TRY_CALL(m_pimpl_->m_patches_.erase(m->id()));
}


DataBlock const *Attribute::at(const MeshBlock *m) const
{
    ASSERT(m_pimpl_ != nullptr);

    try { return m_pimpl_->m_patches_.at(m->id()).get(); }
    catch (std::out_of_range const &err)
    {
        throw std::out_of_range(
                FILE_LINE_STAMP_STRING + "Can not find Mesh Block! "
                        "[ null mesh: " + string_cast(m == nullptr) + "]");

    }
}

DataBlock *Attribute::at(const MeshBlock *m)
{
    ASSERT(m_pimpl_ != nullptr);

    try { return m_pimpl_->m_patches_.at(m->id()).get(); }
    catch (std::out_of_range const &err)
    {
        throw std::out_of_range(
                FILE_LINE_STAMP_STRING + "Can not find Mesh Block! "
                        "[ null mesh: " + string_cast(m == nullptr) + "]");

    }
}

void Attribute::insert(const MeshBlock *m, const std::shared_ptr<DataBlock> &d)
{
    ASSERT(m_pimpl_ != nullptr);
    if (m != nullptr || d == nullptr) { m_pimpl_->m_patches_.emplace(std::make_pair(m->id(), d)); }
    else { WARNING << " try to insert null mesh or data block" << std::endl; }
}


}}//namespace simpla { namespace mesh
