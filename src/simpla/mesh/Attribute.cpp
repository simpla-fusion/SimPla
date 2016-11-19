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

    std::map<std::type_index, std::function<std::shared_ptr<DataBlock>(std::shared_ptr<MeshBlock> const &,
                                                                       void *)> > m_data_factory;
};

Attribute::Attribute(std::string const &s, std::string const &config_str)
        : Object(), m_name_(s), m_pimpl_(new pimpl_s)
{
    db["config"] = config_str;
}

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

bool Attribute::has(std::shared_ptr<MeshBlock> const &m) const
{
    ASSERT(m_pimpl_ != nullptr);
    return (m == nullptr) ? false : m_pimpl_->m_patches_.find(m->id()) != m_pimpl_->m_patches_.end();
}


void Attribute::erase(std::shared_ptr<MeshBlock> const &m)
{
    ASSERT(m_pimpl_ != nullptr);
    TRY_CALL(m_pimpl_->m_patches_.erase(m->id()));
}


std::shared_ptr<DataBlock> const &Attribute::at(std::shared_ptr<MeshBlock> const &m) const
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

std::shared_ptr<DataBlock> &Attribute::at(std::shared_ptr<MeshBlock> const &m)
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

std::shared_ptr<DataBlock> &Attribute::get(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &p)
{
    ASSERT(m_pimpl_ != nullptr);
    if (!has(m)) { insert(m, p); }
    return at(m);
}


void Attribute::insert(const std::shared_ptr<MeshBlock> &m, const std::shared_ptr<DataBlock> &p)
{
    WARNING << "Wrong Way!" << std::endl;
    ASSERT(m_pimpl_ != nullptr);

    if (m == nullptr) { OUT_OF_RANGE << " try to insert null mesh or data block" << std::endl; }
    else
    {

        if (p != nullptr)
        {
            m_pimpl_->m_patches_.emplace(std::make_pair(m->id(), p));

        } else
        {
            auto it = m_pimpl_->m_data_factory.find(m->typeindex());

            if (it != m_pimpl_->m_data_factory.end())
            {
                m_pimpl_->m_patches_.emplace(std::make_pair(m->id(), it->second(m, nullptr)));

            } else
            {
                RUNTIME_ERROR << " data block factory is not registered!" << std::endl;
            }

        }
    }
}


void Attribute::register_data_block_factory(
        std::type_index idx,
        const std::function<std::shared_ptr<DataBlock>(std::shared_ptr<MeshBlock> const &, void *)> &f)
{
    m_pimpl_->m_data_factory[idx] = f;

};
}}//namespace simpla { namespace mesh
