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

struct AttributeBase::pimpl_s { std::map<id_type, std::shared_ptr<DataBlock>> m_patches_; };

AttributeBase::AttributeBase(std::string const &config_str) : Object(), m_pimpl_(new pimpl_s) { db.parse(config_str); }

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


void AttributeBase::load(const data::DataEntityTable &) { UNIMPLEMENTED; }

void AttributeBase::save(data::DataEntityTable *) const { UNIMPLEMENTED; }

const DataBlock *AttributeBase::find(const id_type &m) const
{
    ASSERT(m_pimpl_ != nullptr);
    auto it = m_pimpl_->m_patches_.find(m);

    return (it != m_pimpl_->m_patches_.end()) ? it->second.get() : nullptr;
}

DataBlock *AttributeBase::find(const id_type &m)
{
    ASSERT(m_pimpl_ != nullptr);
    auto it = m_pimpl_->m_patches_.find(m);

    return (it != m_pimpl_->m_patches_.end()) ? it->second.get() : nullptr;
}

void AttributeBase::erase(const id_type &m)
{
    ASSERT(m_pimpl_ != nullptr);
    TRY_CALL(m_pimpl_->m_patches_.erase(m));
}


std::shared_ptr<DataBlock> const &AttributeBase::at(const id_type &m) const
{
    ASSERT(m_pimpl_ != nullptr);
    return m_pimpl_->m_patches_.at(m);
//    auto it = m_pimpl_->m_patches_.find(m);
//    if (it != m_pimpl_->m_patches_.end())
//    {
//        return it->second;
//    } else
//    {
//        throw std::out_of_range(
//                FILE_LINE_STAMP_STRING + "Can not find Mesh Block! "
//                        "[ null mesh: " + string_cast(m) + "]");
//
//    }
}

std::shared_ptr<DataBlock> &AttributeBase::at(const id_type &m)
{
    ASSERT(m_pimpl_ != nullptr);
    return m_pimpl_->m_patches_.at(m);
//    auto it = m_pimpl_->m_patches_.find(m);
//    if (it != m_pimpl_->m_patches_.end())
//    {
//        return it->second;
//    } else
//    {
//        throw std::out_of_range(
//                FILE_LINE_STAMP_STRING + "Can not find Mesh Block! "
//                        "[ null mesh: " + string_cast(m) + "]");
//
//    }
}

std::pair<std::shared_ptr<DataBlock>, bool>
AttributeBase::emplace(const id_type &m, const std::shared_ptr<DataBlock> &p)
{
    if (p != nullptr && m_pimpl_ != nullptr)
    {
        auto res = m_pimpl_->m_patches_.emplace(m, p);

        return std::make_pair(res.first->second, res.second);

    } else
    {
        return std::make_pair(nullptr, false);
    }

}


AttributeView::AttributeView(std::shared_ptr<AttributeBase> const &attr)
        : m_attr_(attr)
{
};

AttributeView::AttributeView(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d,
                             std::shared_ptr<AttributeBase> const &attr)
        : m_attr_(attr), m_mesh_block_holder_(m), m_data_block_holder_(d)
{
};

AttributeView::~AttributeView() { destroy(); }

void AttributeView::deploy() {}

void AttributeView::destroy() {}

void AttributeView::pre_process()
{
    if (is_valid()) { return; } else { concept::LifeControllable::pre_process(); }

    if (m_mesh_block_holder_ != nullptr) { m_mesh_block_ = m_mesh_block_holder_.get(); }

    ASSERT(m_mesh_block_ != nullptr);

    if (m_data_block_holder_ != nullptr) { m_data_block_ = m_data_block_holder_.get(); }
    else if (m_attr_ != nullptr) { m_data_block_ = m_attr_->find(m_mesh_block_->id()); }

    if (m_data_block_ == nullptr)
    {
        m_data_block_holder_ = create_data_block(m_mesh_block_, nullptr);
        if (m_attr_ != nullptr) { m_attr_->emplace(m_mesh_block_->id(), m_data_block_holder_); }
        m_data_block_ = m_data_block_holder_.get();
    }
    ASSERT(m_data_block_ != nullptr);

    m_data_block_->pre_process();


}

void AttributeView::post_process()
{
    if (!is_valid()) { return; } else { concept::LifeControllable::post_process(); }
    m_data_block_ = nullptr;
    m_mesh_block_ = nullptr;
    m_data_block_holder_.reset();
    m_mesh_block_holder_.reset();
}


void AttributeView::move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d)
{
    if (m == nullptr || (m == m_mesh_block_holder_ && d == m_data_block_holder_)) { return; }

    post_process();

    m_mesh_block_holder_ = m;
    m_data_block_holder_ = d;
}

void AttributeView::initialize()
{
    pre_process();
    auto *p = data();
    if (p != nullptr) p->clear();
}

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
