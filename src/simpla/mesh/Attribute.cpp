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

std::string AttributeView::name() const { return m_attr_->name(); };


std::ostream &AttributeView::print(std::ostream &os, int indent) const
{
    if (m_mesh_ != nullptr && m_data_ != nullptr) { m_data_->print(os, indent + 1); } else { os << "not-deployed!"; }
    return os;
}


void AttributeView::move_to(MeshBlock const *m)
{
    ASSERT (m != nullptr)
    if (m != m_mesh_)
    {
        m_mesh_ = m;

        try
        {
            m_data_ = m_attr_->at(m_mesh_);

        }
        catch (std::out_of_range const &error)
        {
            auto res = clone(m_mesh_);
            m_attr_->insert(m_mesh_, res);
            m_data_ = res.get();
//            throw std::runtime_error(FILE_LINE_STAMP_STRING + " Cannot clone data block [m_data_==null]");
        }
        //        else if (m_data_ != nullptr)
//        {
//            VERBOSE << " Create data block of [" << name() << "] from " << m_mesh_->id() << " to " << m->id()
//                    << std::endl;
//            auto res = m_data_->clone(m);
//            m_data_ = res.get();
//            m_attr_->insert(m_mesh_, res);
//            m_mesh_ = m;
//
//        }
    }

};

void AttributeView::deploy()
{
    if (m_data_ == nullptr) { move_to(m_mesh_); }
    m_data_->deploy();
}


void AttributeView::destroy() { if (m_data_ != nullptr) { m_data_->destroy(); }};

void AttributeView::erase()
{
    ASSERT (m_attr_ != nullptr);

    m_attr_->erase(m_mesh_);
    m_mesh_ = nullptr;
}

void AttributeView::sync(MeshBlock const *other, bool only_ghost)
{
    try { m_data_->sync(m_attr_->at(other), only_ghost); } catch (std::out_of_range const &) {}

}


}}//namespace simpla { namespace mesh
