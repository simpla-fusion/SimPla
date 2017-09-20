//
// Created by salmon on 17-8-11.
//

#ifndef SIMPLA_PARTICLEDATA_H
#define SIMPLA_PARTICLEDATA_H

#include "simpla/SIMPLA_config.h"

#include <simpla/data/DataBlock.h>

namespace simpla {
struct ParticleData : public data::DataEntity {
    SP_DEFINE_FANCY_TYPE_NAME(ParticleData, data::DataEntity);
    ParticleData(int DOF = 0, size_type NumberOfPIC = 100) : m_dof_(DOF), m_number_of_pic_(NumberOfPIC) {}
    ~ParticleData() override = default;

    size_type GetNumberOfPIC() const { return m_number_of_pic_; }
    void SetNumberOfPIC(size_type n) { m_number_of_pic_ = n; }

   private:
    int m_dof_ = 0;
    size_type m_number_of_pic_ = 100;
};
}  // namespace simpla

#endif  // SIMPLA_PARTICLEDATA_H
