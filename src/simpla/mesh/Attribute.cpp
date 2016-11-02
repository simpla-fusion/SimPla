//
// Created by salmon on 16-10-20.
//


#include "Attribute.h"

namespace simpla { namespace mesh
{
struct AttributeBase::pimpl_s
{
    std::shared_ptr<Atlas> m_atlas_;

    std::map<mesh_id_type, std::shared_ptr<PatchBase>> m_patches_;

    typename toolbox::Object::id_type m_id_;
};

AttributeBase::AttributeBase() : m_pimpl_(new pimpl_s)
{
    m_pimpl_->m_atlas_ = std::make_shared<Atlas>();
}

AttributeBase::AttributeBase(std::shared_ptr<MeshBlock> const &m) : m_pimpl_(new pimpl_s)
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
    {        item.second->print(os, indent + 1);    }
    return os;
}

void AttributeBase::deploy()
{
    if (m_pimpl_->m_id_ == 0)
    {
        assert(m_pimpl_->m_atlas_->mesh() != nullptr);
        m_pimpl_->m_id_ = m_pimpl_->m_atlas_->mesh()->id();
    }

    patch()->deploy();

    update();
}

void AttributeBase::move_to(mesh_id_type id)
{
    m_pimpl_->m_id_ = id;
    update();

}

AttributeBase::mesh_id_type AttributeBase::mesh_id() const { return m_pimpl_->m_id_; }


std::shared_ptr<Atlas> const &AttributeBase::atlas() const { return m_pimpl_->m_atlas_; };

bool AttributeBase::has(const mesh_id_type &t_id)
{
    return m_pimpl_->m_patches_.find(t_id) != m_pimpl_->m_patches_.end();
};

MeshBlock const *AttributeBase::mesh(mesh_id_type t_id) const
{
    return m_pimpl_->m_atlas_->mesh(t_id);
}


PatchBase const *AttributeBase::patch(mesh_id_type t_id) const
{
    return m_pimpl_->m_patches_.at((t_id == 0) ? mesh()->id() : t_id).get();
}

PatchBase *AttributeBase::patch(mesh_id_type t_id)
{
    PatchBase *res;

    t_id = (t_id == 0) ? mesh()->id() : t_id;

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


}}