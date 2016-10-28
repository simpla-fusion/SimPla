//
// Created by salmon on 16-10-20.
//


#include "Attribute.h"

namespace simpla { namespace mesh
{
struct AttributeBase::pimpl_s
{
    std::shared_ptr<Atlas> m_atlas_;
    std::map<id_type, std::shared_ptr<PatchBase>> m_patches_;
    MeshBase const *m_mesh_;
    PatchBase *m_patch_;
    typename toolbox::Object::id_type m_id_;
};

AttributeBase::AttributeBase() : m_pimpl_(new pimpl_s)
{
    m_pimpl_->m_mesh_ = nullptr;
    m_pimpl_->m_patch_ = nullptr;
    m_pimpl_->m_atlas_ = std::make_shared<Atlas>();
}

AttributeBase::AttributeBase(std::shared_ptr<MeshBase> const &m) : m_pimpl_(new pimpl_s)
{
    m_pimpl_->m_atlas_ = std::make_shared<Atlas>();
    m_pimpl_->m_atlas_->add(m);
    m_pimpl_->m_id_ = m->id();
};

AttributeBase::AttributeBase(std::shared_ptr<Atlas> const &m) : m_pimpl_(new pimpl_s) { m_pimpl_->m_atlas_ = m; }

AttributeBase::AttributeBase(AttributeBase &&other) : m_pimpl_(std::move(other.m_pimpl_)) {}

AttributeBase::~AttributeBase() {}

std::ostream &AttributeBase::print(std::ostream &os, int indent) const
{
    for (auto const &item:m_pimpl_->m_patches_)
    {
        item.second->print(os, indent + 1);
    }
    return os;
}

void AttributeBase::deploy()
{
    if (m_pimpl_->m_id_ == 0)
    {
        assert(m_pimpl_->m_atlas_->first() != nullptr);
        m_pimpl_->m_id_ = m_pimpl_->m_atlas_->first()->id();
    }

    m_pimpl_->m_mesh_ = mesh(m_pimpl_->m_id_);
    m_pimpl_->m_patch_ = patch(m_pimpl_->m_id_);
    assert(m_pimpl_->m_patch_ != nullptr);
    m_pimpl_->m_patch_->deploy();
}

AttributeBase::id_type const &AttributeBase::mesh_id() const { return m_pimpl_->m_id_; }

std::shared_ptr<Atlas> AttributeBase::atlas() const { return m_pimpl_->m_atlas_; };

bool AttributeBase::has(const id_type &t_id) { return m_pimpl_->m_patches_.find(t_id) != m_pimpl_->m_patches_.end(); };

void AttributeBase::move_to(const id_type &t_id) { m_pimpl_->m_id_ = t_id; };


PatchBase const *AttributeBase::patch(id_type const &t_id) const
{
    return m_pimpl_->m_patches_.at(t_id).get();
}

PatchBase *AttributeBase::patch(id_type const &t_id)
{
    PatchBase *res;

    assert(t_id != 0);
    auto it = m_pimpl_->m_patches_.find(t_id);

    if (it != m_pimpl_->m_patches_.end())
    {
        res = it->second.get();
    } else
    {
        auto p = create(t_id);
        res = m_pimpl_->m_patches_.emplace(t_id, p).first->second.get();
    }

    return res;
};

PatchBase *AttributeBase::patch()
{
    assert(m_pimpl_->m_patch_ != nullptr);
    return m_pimpl_->m_patch_;
};

PatchBase const *AttributeBase::patch() const
{
    assert(m_pimpl_->m_patch_ != nullptr);
    return m_pimpl_->m_patch_;
};

MeshBase const *AttributeBase::mesh() const
{
    assert(m_pimpl_->m_mesh_ != nullptr);
    return m_pimpl_->m_mesh_;
};

MeshBase const *AttributeBase::mesh(id_type const &t_id) const
{
    return m_pimpl_->m_atlas_->at(t_id).get();
}

}}