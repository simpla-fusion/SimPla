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
    typedef typename Atlas::id_type id_type;


    AttributeBase();

    AttributeBase(AttributeBase const &) = delete;

    AttributeBase(AttributeBase &&);

    virtual ~AttributeBase();

    virtual std::shared_ptr<PatchBase> create(id_type id)=0;

    virtual void move_to(id_type t_id);

    id_type mesh_id() const;

    bool has(id_type t_id);

    PatchBase *get(id_type t_id);

    MeshBase const *mesh() const;

    PatchBase *patch();

    PatchBase const *patch() const;

    template<typename TM> TM const *mesh_as() const
    {
        MeshBase const *res = mesh();
        assert(res->is_a<TM>());
        return static_cast<TM const *>(res);
    }

    template<typename T> T *patch_as()
    {
        PatchBase const *res = patch();
        assert(res->is_a<T>());
        return static_cast<T *>(res);
    }

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;


};

template<typename V, typename M, MeshEntityType IFORM = VERTEX>
class Attribute : public AttributeBase
{
public:
    static constexpr MeshEntityType iform = IFORM;
    typedef V value_type;
    typedef M mesh_type;
    typedef Patch<value_type, mesh_type, iform> patch_type;


    virtual std::shared_ptr<PatchBase>
    AttributeBase::create(id_type id)
    {
        return std::make_shared<patch_type>(mesh_as<mesh_type>(id));
    }

private:

};
}}
#endif //SIMPLA_ATTRIBUTE_H
