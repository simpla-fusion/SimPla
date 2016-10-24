//
// Created by salmon on 16-10-24.
//

#ifndef SIMPLA_WRAPPER_H
#define SIMPLA_WRAPPER_H

#include <memory>
#include "../Attribute.h"
#include "../Atlas.h"

namespace simpla { namespace mesh
{


namespace detail
{
std::shared_ptr<AttributeBase>
create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, MeshEntityType const &,
                      std::shared_ptr<Atlas> const &m, std::string const &name);


std::shared_ptr<AttributeBase>
create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, MeshEntityType const &,
                      std::shared_ptr<MeshBase> const &m, std::string const &name);

std::shared_ptr<PatchBase>
create_patch_impl(std::type_info const &type_info, std::type_info const &mesh_info, MeshEntityType const &,
                  std::shared_ptr<MeshBase> const &m);

}

template<typename TV, typename TM, MeshEntityType IFORM, typename ...Args>
std::shared_ptr<Attribute<Patch<TV, TM, IFORM> > > create_attribute(Args &&...args)
{
    return std::dynamic_pointer_cast<Attribute<Patch<TV, TM, IFORM> >>(detail::create_attribute_impl(
            typeid(TV), typeid(TM), IFORM, std::forward<Args>(args)...));;
};

template<typename TV, typename TM, MeshEntityType IFORM, typename ...Args>
std::shared_ptr<Patch<TV, TM, IFORM> > create_patch(Args &&...args)
{
    return std::dynamic_pointer_cast<Patch<TV, TM, IFORM> >(detail::create_patch_impl(
            typeid(TV), typeid(TM), IFORM, std::forward<Args>(args)...));;
};
}}
#endif //SIMPLA_WRAPPER_H
