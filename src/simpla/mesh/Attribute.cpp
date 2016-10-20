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
    id_type m_id_;
    MeshBase const *m_mesh_;
    PatchBase *m_patch_;
};

AttributeBase::AttributeBase(std::shared_ptr<MeshBase> const &m) : m_pimpl_(new pimpl_s)
{
    m_pimpl_->m_atlas_ = std::make_shared<Atlas>();
    m_pimpl_->m_atlas_->add(m);
    move_to(m->id());
};

AttributeBase::AttributeBase(std::shared_ptr<Atlas> const &m) : m_pimpl_(new pimpl_s)
{
    m_pimpl_->m_atlas_ = m;

    auto p = m_pimpl_->m_atlas_->first();

    if (p != nullptr)
    {
        move_to(p->id());
    } else
    {
        m_pimpl_->m_mesh_ = nullptr;
        m_pimpl_->m_patch_ = nullptr;
    }

}

AttributeBase::AttributeBase(AttributeBase &&other) : m_pimpl_(std::move(other.m_pimpl_)) {}

AttributeBase::~AttributeBase() {}

AttributeBase::id_type const &AttributeBase::mesh_id() const { return m_pimpl_->m_id_; }

std::shared_ptr<Atlas> AttributeBase::atlas() const { return m_pimpl_->m_atlas_; };

bool AttributeBase::has(const id_type &t_id) { return m_pimpl_->m_patches_.find(t_id) != m_pimpl_->m_patches_.end(); };

void AttributeBase::move_to(const id_type &t_id)
{
    m_pimpl_->m_mesh_ = m_pimpl_->m_atlas_->at(t_id).get();
    m_pimpl_->m_patch_ = patch(t_id);
    m_pimpl_->m_id_ = t_id;
};

PatchBase const *AttributeBase::patch(id_type const &t_id) const
{
    return m_pimpl_->m_patches_.at(t_id).get();
}

PatchBase *AttributeBase::patch(id_type const &t_id)
{
    PatchBase *res;

    auto it = m_pimpl_->m_patches_.find(t_id);

    if (it != m_pimpl_->m_patches_.end())
    {
        res = it->second.get();
    } else
    {
        res = m_pimpl_->m_patches_.emplace(t_id, create(t_id)).first->second.get();
    }

    return res;
};

PatchBase *AttributeBase::patch() { return m_pimpl_->m_patch_; };

PatchBase const *AttributeBase::patch() const { return m_pimpl_->m_patch_; };

MeshBase const *AttributeBase::mesh() const { return m_pimpl_->m_mesh_; };

MeshBase const *AttributeBase::mesh(id_type const &t_id) const
{
    return m_pimpl_->m_atlas_->at(t_id).get();
}

}}