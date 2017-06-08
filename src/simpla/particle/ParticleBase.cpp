//
// Created by salmon on 17-6-8.
//
#include "ParticleBase.h"
namespace simpla {

struct ParticleBase::pimpl_s {
    int m_dof_ = 7;

    std::multimap<EntityId, std::shared_ptr<bucket_holder>> m_data_;
};
}