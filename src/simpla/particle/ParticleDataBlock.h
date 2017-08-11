//
// Created by salmon on 17-8-11.
//

#ifndef SIMPLA_PARTICLEDATABLOCK_H
#define SIMPLA_PARTICLEDATABLOCK_H

#include <simpla/data/DataBlock.h>

namespace simpla {
struct ParticleDataBlock : public data::DataBlock {
    SP_OBJECT_HEAD(ParticleDataBlock, data::DataBlock);
    ParticleDataBlock(int DOF = 0) : m_dof_(DOF) {}
    ~ParticleDataBlock() override = default;

   private:
    int m_dof_ = 0;
};
}  // namespace simpla

#endif  // SIMPLA_PARTICLEDATABLOCK_H
