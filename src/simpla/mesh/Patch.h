//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Serializable.h>
#include <simpla/toolbox/Printable.h>

namespace simpla { namespace mesh
{

struct PatchBase : public toolbox::Serializable, public toolbox::Printable
{
    virtual bool is_a(std::type_info const &info) const { return info == typeid(PatchBase); };

    virtual void deploy() {};

    virtual MeshEntityType entity_type() const =0;
};

template<typename V, MeshEntityType IFORM>
class Patch : public PatchBase
{
    typedef PatchBase base_type;
    typedef Patch<V, IFORM> this_type;
public:
    virtual bool is_a(std::type_info const &info) const { return info == typeid(this_type) || base_type::is_a(info); };

    virtual MeshEntityType entity_type() const { return IFORM; };
};
}}
#endif //SIMPLA_PATCH_H
