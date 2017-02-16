//
// Created by salmon on 17-2-16.
//

#ifndef SIMPLA_MANAGER_H
#define SIMPLA_MANAGER_H

#include <simpla/SIMPLA_config.h>
#include <map>
#include "DomainView.h"
#include "Patch.h"
namespace simpla {
namespace engine {
class Manager {
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;
    std::map<id_type, std::shared_ptr<DomainView>> m_views_;
};
template <typename U>
struct ManagerAdapter : public Manager, public U {};
}  // namespace engine{
}  // namespace simpla{

#endif  // SIMPLA_MANAGER_H
