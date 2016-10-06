//
// Created by salmon on 16-6-13.
//

#ifndef SIMPLA_PARTICLEBASE_H
#define SIMPLA_PARTICLEBASE_H

#include "../sp_config.h"

namespace simpla { namespace particle
{

struct ParticleBase
{
    const size_type m_ele_size_in_byte_;

    struct spPagePool m_pool_;

    struct spPage **m_data_;
};

}}//namespace simpla{namespace particle{
#endif //SIMPLA_PARTICLEBASE_H
