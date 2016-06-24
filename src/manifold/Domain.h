//
// Created by salmon on 16-6-24.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include "Manifold.h"

namespace simpla
{
class SimPlaDomain
{
    manifold::Atlas m_atlas_;

    std::map<mesh::MeshBlockId, std::shared_ptr<ProblemDomain>> m_sub_domains_;

};
}
#endif //SIMPLA_DOMAIN_H
