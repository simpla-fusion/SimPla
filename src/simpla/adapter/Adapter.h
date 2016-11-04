//
// Created by salmon on 16-10-24.
//

#ifndef SIMPLA_WRAPPER_H
#define SIMPLA_WRAPPER_H

#include <memory>

#include <simpla/mesh/Atlas.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/simulation/Context.h>

namespace simpla
{


namespace detail
{
std::shared_ptr<mesh::Attribute>
create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
                      std::shared_ptr<mesh::Atlas> const &m, std::string const &name);


std::shared_ptr<mesh::Attribute>
create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
                      std::shared_ptr<mesh::MeshBlock> const &m, std::string const &name);

std::shared_ptr<mesh::DataBlock>
create_patch_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
                  std::shared_ptr<mesh::MeshBlock> const &m);

}

std::shared_ptr<mesh::Atlas> create_atlas(std::string const &name);

template<typename TV, typename TM, mesh::MeshEntityType IFORM, typename ...Args>
std::shared_ptr<mesh::Attribute<mesh::DataBlock<TV, TM, IFORM> > > create_attribute(Args &&...args)
{
    return std::dynamic_pointer_cast<mesh::Attribute<mesh::DataBlock<TV, TM, IFORM> >>(detail::create_attribute_impl(
            typeid(TV), typeid(TM), IFORM, std::forward<Args>(args)...));;
};

template<typename TV, typename TM, mesh::MeshEntityType IFORM, typename ...Args>
std::shared_ptr<mesh::DataBlock<TV, TM, IFORM> > create_patch(Args &&...args)
{
    return std::dynamic_pointer_cast<mesh::DataBlock<TV, TM, IFORM> >(detail::create_patch_impl(
            typeid(TV), typeid(TM), IFORM, std::forward<Args>(args)...));;
};

std::shared_ptr<simulation::ContextBase> create_context(std::string const &name);

}
#endif //SIMPLA_WRAPPER_H
