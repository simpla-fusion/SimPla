//
// Created by salmon on 16-10-20.
//


#include "Attribute.h"

namespace simpla { namespace mesh
{
struct AttributeBase::pimpl_s
{
    id_type m_id_;
    MeshBase const *m_mesh_;
    PatchBase *m_patch_;
    std::shared_ptr<Atlas> m_atlas_;
    std::map<id_type, std::shared_ptr<PatchBase>> m_patches_;
};

AttributeBase::AttributeBase() : m_pimpl_(new pimpl_s) {}

AttributeBase::AttributeBase(AttributeBase &&other)
        : m_pimpl_(std::move(other.m_pimpl_)) {}

AttributeBase::~AttributeBase() {}

id_type AttributeBase::mesh_id() const { return m_pimpl_->m_id_; }

bool AttributeBase::has(id_type t_id) { return m_pimpl_->m_patches_.find(t_id) != m_pimpl_->m_patches_.end(); };

void AttributeBase::move_to(id_type t_id)
{
    m_pimpl_->m_mesh_ = m_pimpl_->m_atlas_->get(t_id).get();
    m_pimpl_->m_patch_ = get(t_id);
    m_pimpl_->m_id_ = t_id;
};


PatchBase *AttributeBase::get(id_type t_id)
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

MeshBase const *AttributeBase::mesh() const { return m_pimpl_->m_mesh_; };

PatchBase *AttributeBase::patch() { return m_pimpl_->m_patch_; };

PatchBase const *AttributeBase::patch() const { return m_pimpl_->m_patch_; };


}}