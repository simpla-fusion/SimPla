//
// Created by salmon on 17-6-5.
//

#ifndef SIMPLA_ICRFANTENNA_H
#define SIMPLA_ICRFANTENNA_H

#include "simpla/SIMPLA_config.h"

#include <cmath>
#include <memory>

#include "simpla/algebra/Algebra.h"
#include "simpla/data/DataEntry.h"
#include "simpla/engine/Engine.h"
#include "simpla/physics/Field.h"
#include "simpla/physics/PhysicalConstants.h"

namespace simpla {

using namespace data;

template <typename TDomain>
class ICRFAntenna : public TDomain {
    SP_DOMAIN_HEAD(ICRFAntenna, TDomain);

    Field<this_type, Real, EDGE> J{this, "name"_ = "J"};

    SP_PROPERTY(Vec3, Amplify) = {1, 0, 0};
    SP_PROPERTY(Real, Frequency) = 1.0;
    SP_PROPERTY(Vec3, WaveNumber) = {1, 0, 0};
};

template <typename TM>
void ICRFAntenna<TM>::DoInitialCondition(Real time_now) {
    //    m_domain_->GetMesh()->SetEmbeddedBoundary(m_domain_->GetName(), m_domain_->GetGeoBody());
}

template <typename TM>
void ICRFAntenna<TM>::DoAdvance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST

    SP_CMD((J = [=](point_type const& x) -> nTuple<Real, 3> {
        nTuple<Real, 3> res = m_Amplify_ * std::sin(m_WaveNumber_[0] * x[0] + m_WaveNumber_[1] * x[1] +
                                                    m_WaveNumber_[2] * x[2] + TWOPI * m_Frequency_ * time_now);
        return res;
    }));
}

}  // namespace simpla;
#endif  // SIMPLA_ICRFANTENNA_H
