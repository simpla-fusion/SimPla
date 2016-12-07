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


AttributeView::AttributeView(std::shared_ptr<AttributeBase> const &attr) : m_attr_(attr) {};

//AttributeView::AttributeView(AttributeView const &other) :
//        m_id_(other.m_id_), m_data_(other.m_data_), m_attr_(other.m_attr_)
//{
//
//}
//
//AttributeView::AttributeView(AttributeView &&other) :
//        m_id_(other.m_id_), m_data_(std::move(other.m_data_)), m_attr_(std::move(other.m_attr_))
//{
//}
//
//void AttributeView::swap(AttributeView &other)
//{
//    std::swap(m_id_, other.m_id_);
//    std::swap(m_data_, other.m_data_);
//    std::swap(m_attr_, other.m_attr_);
//}

AttributeView::~AttributeView() {}

id_type AttributeView::mesh_id() const { return is_valid() ? m_mesh_block_->id() : 0; }

std::shared_ptr<AttributeBase> &AttributeView::attribute() { return m_attr_; }

std::shared_ptr<AttributeBase> const &AttributeView::attribute() const { return m_attr_; }

DataBlock *AttributeView::data_block() { return m_data_.get(); };

DataBlock const *AttributeView::data_block() const { return m_data_.get(); };

void AttributeView::move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d)
{
    if (m == nullptr || m == m_mesh_block_) { return; }
    post_process();
    m_mesh_block_ = m;
    m_data_ = d;
}


void AttributeView::pre_process()
{
    if (is_valid()) { return; } else { concept::LifeControllable::pre_process(); }

    ASSERT(m_mesh_block_ != nullptr);
    if (m_data_ != nullptr) { return; }
    else if (m_attr_ != nullptr && m_attr_->find(m_mesh_block_->id())) { m_data_ = m_attr_->at(m_mesh_block_->id()); }
    else
    {
        m_data_ = create_data_block(m_mesh_block_, nullptr);
        m_data_->pre_process();
    }
    ASSERT(m_data_ != nullptr);
}

void AttributeView::post_process()
{
    if (!is_valid()) { return; } else { concept::LifeControllable::post_process(); }
    m_data_.reset();
    m_mesh_block_.reset();
}


void AttributeView::clear()
{
    pre_process();
    m_data_->clear();
}


}
}//namespace simpla { namespace mesh
