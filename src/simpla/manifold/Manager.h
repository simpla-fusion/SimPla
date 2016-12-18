//
// Created by salmon on 16-12-8.
//

#ifndef SIMPLA_MANAGER_H
#define SIMPLA_MANAGER_H

#include <simpla/concept/Configurable.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/LifeControllable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/model/Model.h>
#include "simpla/mesh/Worker.h"
#include "simpla/mesh/Atlas.h"
#include "simpla/mesh/Patch.h"

namespace simpla { namespace mesh
{

class Manager :
        public concept::Configurable,
        public concept::Printable,
        public concept::Serializable,
        public concept::LifeControllable
{
public:

private:
    mesh::Atlas m_atlas_;
    model::Model m_model_;
    AttributeCollection m_attrs_;
    PatchCollection m_patchs_;
    std::map<std::type_index, std::function<std::shared_ptr<Worker>()> > m_worker_factory_;
};
}} //namespace simpla {namespace mesh{
#endif //SIMPLA_MANAGER_H
