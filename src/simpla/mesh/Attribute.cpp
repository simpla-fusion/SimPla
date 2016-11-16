//
// Created by salmon on 16-10-20.
//

#include <typeindex>
#include <simpla/toolbox/Log.h>
#include "Atlas.h"
#include "Attribute.h"
#include "MeshBlock.h"
#include "DataBlock.h"

namespace simpla { namespace mesh
{
struct Attribute::pimpl_s
{
    std::map<id_type, std::shared_ptr<DataBlock>> m_patches_;

    std::map<std::type_index,
            std::function<std::shared_ptr<DataBlock>(MeshBlock const *, void *p)> > m_data_factory;
};

Attribute::Attribute(std::string const &s) : Object(), m_name_(s), m_pimpl_(new pimpl_s) {}

Attribute::~Attribute() {}

std::ostream &Attribute::print(std::ostream &os, int indent) const
{
    if (m_pimpl_ != nullptr)
    {
        os << std::setw(indent) << " " << std::endl;
        for (auto const &item:m_pimpl_->m_patches_)
        {
//            os << std::setw(indent) << " " << item.second->mesh()->name() << " = {";
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


std::shared_ptr<DataBlock> const &Attribute::at(const MeshBlock *m) const
{
    ASSERT(m_pimpl_ != nullptr);

    auto it = m_pimpl_->m_patches_.find(m->id());
    if (it != m_pimpl_->m_patches_.end())
    {
        return it->second;
    } else
    {
        throw std::out_of_range(
                FILE_LINE_STAMP_STRING + "Can not find Mesh Block! "
                        "[ null mesh: " + string_cast(m == nullptr) + "]");

    }
}

std::shared_ptr<DataBlock> &Attribute::at(const MeshBlock *m)
{
    ASSERT(m_pimpl_ != nullptr);

    auto it = m_pimpl_->m_patches_.find(m->id());
    if (it != m_pimpl_->m_patches_.end())
    {
        return it->second;
    } else
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

std::shared_ptr<DataBlock> Attribute::create_data_block(MeshBlock const *m, void *p) const
{
    auto it = m_pimpl_->m_data_factory.find(m->typeindex());
    if (it != m_pimpl_->m_data_factory.end()) { return it->second(m, p); }
    else { return std::shared_ptr<DataBlock>(nullptr); }
}

void Attribute::register_data_block_factroy(
        std::type_index idx,
        const std::function<std::shared_ptr<DataBlock>(const MeshBlock *, void *)> &f)
{
    m_pimpl_->m_data_factory[idx] = f;

};
}}//namespace simpla { namespace mesh
