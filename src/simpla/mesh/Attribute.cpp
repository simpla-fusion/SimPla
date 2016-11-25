//
// Created by salmon on 16-10-20.
//

#include <typeindex>
#include <simpla/toolbox/Log.h>
#include "simpla/manifold/Atlas.h"
#include "Attribute.h"
#include "MeshBlock.h"
#include "DataBlock.h"

namespace simpla
{
namespace mesh
{

class ChartBase;


struct AttributeBase::pimpl_s
{
    std::map<id_type, std::shared_ptr<DataBlock>> m_patches_;

    std::map<std::type_index, std::function<std::shared_ptr<DataBlock>(std::shared_ptr<MeshBlock> const &,
                                                                       void *)> > m_data_factory;
};


AttributeBase::AttributeBase(std::string const &s, std::string const &config_str)
        : Object(), m_name_(s), m_pimpl_(new pimpl_s)
{
    if (config_str != "") { db["config"] = config_str; }
}

AttributeBase::~AttributeBase() {}

std::ostream &AttributeBase::print(std::ostream &os, int indent) const
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


void AttributeBase::load(const data::DataBase &) { UNIMPLEMENTED; }

void AttributeBase::save(data::DataBase *) const { UNIMPLEMENTED; }

bool AttributeBase::has(const id_type &m) const
{
    ASSERT(m_pimpl_ != nullptr);
    return m_pimpl_->m_patches_.find(m) != m_pimpl_->m_patches_.end();
}


void AttributeBase::erase(const id_type &m)
{
    ASSERT(m_pimpl_ != nullptr);
    TRY_CALL(m_pimpl_->m_patches_.erase(m));
}


std::shared_ptr<DataBlock> const &AttributeBase::at(const id_type &m) const
{
    ASSERT(m_pimpl_ != nullptr);

    auto it = m_pimpl_->m_patches_.find(m);
    if (it != m_pimpl_->m_patches_.end())
    {
        return it->second;
    } else
    {
        throw std::out_of_range(
                FILE_LINE_STAMP_STRING + "Can not find Mesh Block! "
                        "[ null mesh: " + string_cast(m) + "]");

    }
}

std::shared_ptr<DataBlock> &AttributeBase::at(const id_type &m)
{
    ASSERT(m_pimpl_ != nullptr);

    auto it = m_pimpl_->m_patches_.find(m);
    if (it != m_pimpl_->m_patches_.end())
    {
        return it->second;
    } else
    {
        throw std::out_of_range(
                FILE_LINE_STAMP_STRING + "Can not find Mesh Block! "
                        "[ null mesh: " + string_cast(m) + "]");

    }
}

std::shared_ptr<DataBlock> &AttributeBase::get(id_type const &m, std::shared_ptr<DataBlock> const &p)
{
    ASSERT(m_pimpl_ != nullptr);
    if (!has(m)) { insert_or_assign(m, p); }
    return at(m);
}


std::shared_ptr<DataBlock> AttributeBase::insert_or_assign(const id_type &m, const std::shared_ptr<DataBlock> &p)
{
    ASSERT(m_pimpl_ != nullptr);
    m_pimpl_->m_patches_.emplace(std::make_pair(m, p));
    return p;
}

std::shared_ptr<DataBlock>
AttributeBase::insert_or_assign(const std::shared_ptr<MeshBlock> &m, const std::shared_ptr<DataBlock> &p)
{

    ASSERT(m_pimpl_ != nullptr);

    if (m == nullptr) { OUT_OF_RANGE << " try to insert null mesh or data_block block" << std::endl; }
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
                RUNTIME_ERROR << " data_block block factory is not registered!" << std::endl;
            }

        }
    }
    return p;
}

void AttributeBase::register_data_block_factory(
        std::type_index idx,
        const std::function<std::shared_ptr<DataBlock>(std::shared_ptr<MeshBlock> const &, void *)> &f)
{
    m_pimpl_->m_data_factory[idx] = f;
};


AttributeViewBase::AttributeViewBase(std::shared_ptr<AttributeBase> const &attr) : m_attr_(attr) {};

AttributeViewBase::~AttributeViewBase() {}

id_type AttributeViewBase::mesh_id() const { return m_id_; }

std::shared_ptr<AttributeBase> &AttributeViewBase::attribute() { return m_attr_; }

std::shared_ptr<AttributeBase> const &AttributeViewBase::attribute() const { return m_attr_; }

DataBlock *AttributeViewBase::data_block() { return m_data_.get(); };

DataBlock const *AttributeViewBase::data_block() const { return m_data_.get(); };

void AttributeViewBase::move_to(id_type const &id, std::shared_ptr<DataBlock> const &d)
{
    m_id_ = id;

    if (d != nullptr)
    {
        m_data_id_ = id;
        m_data_ = d;
    }
    deploy();

}

void AttributeViewBase::move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d)
{
    move_to(m->id(), d);
}


void AttributeViewBase::deploy()
{
    if (m_data_id_ != m_id_)
    {
        if (m_attr_ != nullptr && m_attr_->has(m_id_))
        {
            m_data_ = m_attr_->get(m_id_);
            m_data_id_ = m_id_;
        } else
        {
            RUNTIME_ERROR << "empty attribute" << std::endl;
        }
    }

    ASSERT(m_data_ != nullptr);
    m_data_->deploy();
};

void AttributeViewBase::clear()
{
    deploy();
    m_data_->clear();
}

void AttributeViewBase::destroy()
{
    m_data_.reset();
    if (m_attr_ != nullptr) { m_attr_->erase(m_data_id_); }
    m_data_id_ = 0;
}
}
}//namespace simpla { namespace mesh
