//
// Created by salmon on 17-6-5.
//

#ifndef SIMPLA_ICRFANTENNA_H
#define SIMPLA_ICRFANTENNA_H

#include "simpla/SIMPLA_config.h"

#include <cmath>
#include <memory>

#include "simpla/algebra/Algebra.h"
#include "simpla/data/DataNode.h"
#include "simpla/engine/Engine.h"
#include "simpla/physics/PhysicalConstants.h"

namespace simpla {

using namespace data;

template <typename TDomain>
class ICRFAntenna : public TDomain {
    SP_DOMAIN_HEAD(ICRFAntenna, TDomain);

    Field<this_type, Real, EDGE> J{this, "name"_ = "J"};

    Vec3 m_amplify_{0, 0, 0};
    Real m_f_ = 1.0;
    Vec3 m_k_{0, 0, 0};
};

template <typename TM>
std::shared_ptr<data::DataNode> ICRFAntenna<TM>::Serialize() const {
    auto res = data::DataNode::New();
    res->SetValue("Amplify", m_amplify_);
    res->SetValue("Frequency", m_f_);
    res->SetValue("WaveNumber", m_k_);
    return res;
};

template <typename TM>
void ICRFAntenna<TM>::DoInitialCondition(Real time_now) {
    //    m_domain_->GetMesh()->SetEmbeddedBoundary(m_domain_->GetName(), m_domain_->GetGeoBody());
}
template <typename TM>
void ICRFAntenna<TM>::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    m_amplify_ = cfg->GetValue<Vec3>("Amplify", m_amplify_);
    m_f_ = cfg->GetValue<Real>("Frequency", m_f_);
    m_k_ = cfg->GetValue<Vec3>("WaveNumber", m_k_);
}

template <typename TM>
void ICRFAntenna<TM>::DoAdvance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST

    SP_CMD((J = [=](point_type const& x) -> nTuple<Real, 3> {
        nTuple<Real, 3> res =
            m_amplify_ * std::sin(m_k_[0] * x[0] + m_k_[1] * x[1] + m_k_[2] * x[2] + TWOPI * m_f_ * time_now);
        return res;
    }));
}

}  // namespace simpla;
#endif  // SIMPLA_ICRFANTENNA_H
