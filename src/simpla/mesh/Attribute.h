//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include "Patch.h"
#include "Atlas.h"

namespace simpla { namespace mesh
{
/**
 *  Attribute IS-A container of patchs
 */
class AttributeBase : public toolbox::Object
{
public:
    SP_OBJECT_HEAD(AttributeBase, toolbox::Object)

    typedef typename Atlas::id_type id_type;

    AttributeBase();

    AttributeBase(std::shared_ptr<MeshBase> const &);

    AttributeBase(std::shared_ptr<Atlas> const &);

    AttributeBase(AttributeBase const &) = delete;

    AttributeBase(AttributeBase &&);

    virtual ~AttributeBase();

    virtual void deploy();

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual std::shared_ptr<PatchBase> create(id_type const &id) const =0;

    virtual void move_to(const id_type &t_id);

    std::shared_ptr<Atlas> atlas() const;

    bool has(const id_type &t_id);

    id_type const &mesh_id() const;

    virtual MeshBase const *mesh() const;

    virtual MeshBase const *mesh(id_type const &t_id) const;

    virtual PatchBase *patch();

    virtual PatchBase *patch(id_type const &t_id);


    virtual PatchBase const *patch() const;

    virtual PatchBase const *patch(id_type const &t_id) const;

    template<typename T, typename ...Args> T const *mesh_as(Args &&...args) const
    {
        MeshBase const *res = mesh(std::forward<Args>(args)...);
        assert(res->is_a<T>());
        return static_cast<T const *>(res);
    }

//    template<typename T, typename ...Args> T *patch_as(Args &&...args)
//    {
//        PatchBase *res = patch(std::forward<Args>(args)...);
//        assert(res->is_a<T>());
//        return static_cast<T *>(res);
//    }
//
//    template<typename T, typename ...Args> T const *patch_as(Args &&...args) const
//    {
//        PatchBase const *res = patch(std::forward<Args>(args)...);
//        assert(res->is_a<T>());
//        return static_cast<T const *>(res);
//    }

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;


};

template<typename ...> class Attribute;

template<typename TPatch>
class Attribute<TPatch> : public AttributeBase
{
public:

    typedef TPatch patch_type;
    typedef typename patch_type::mesh_type mesh_type;
    typedef typename patch_type::value_type value_type;

    typedef Attribute<TPatch> this_type;

    patch_type *m_patch_ = nullptr;

    MeshEntityType entity_type() const { return patch_type::iform; }

    virtual bool is_a(std::type_info const &t_info) const
    {
        return t_info == typeid(this_type) || AttributeBase::is_a(t_info);
    };


    template<typename ...Args>
    Attribute(Args &&...args) : AttributeBase(std::forward<Args>(args)...) {};

    virtual ~Attribute() {}

    virtual std::shared_ptr<PatchBase> create(id_type const &id) const
    {
        assert(AttributeBase::atlas() != nullptr);
        return std::dynamic_pointer_cast<PatchBase>(
                std::make_shared<patch_type>(
                        AttributeBase::atlas()->template mesh_as<mesh_type>(id).get()));
    }


    patch_type const *patch() const { return static_cast<patch_type const *>(AttributeBase::patch()); }

    patch_type *patch() { return static_cast<patch_type *>(AttributeBase::patch()); }

    virtual void deploy()
    {
        AttributeBase::deploy();
        m_patch_ = patch();
        assert(m_patch_ != nullptr);
    }

    virtual void clear()
    {
        deploy();
        assert(patch() != nullptr);
        patch()->clear();
    }

    inline value_type &get(mesh::MeshEntityId const &s) { return m_patch_->get(s); }

    inline value_type const &get(mesh::MeshEntityId const &s) const { return m_patch_->get(s); }

    inline value_type &operator[](mesh::MeshEntityId const &s) { return get(s); }

    inline value_type const &operator[](mesh::MeshEntityId const &s) const { return get(s); }


    template<typename TOP, typename TRange> void
    apply(TOP const &op, TRange const &r0, this_type const &other)
    {
        deploy();
        m_patch_->apply(op, r0, other.patch());
    }

    template<typename ...Args> void
    apply(Args &&...args)
    {
        deploy();
        m_patch_->apply(std::forward<Args>(args)...);
    }



private:

};
}}
#endif //SIMPLA_ATTRIBUTE_H
