//
// Created by salmon on 17-6-5.
//

#ifndef SIMPLA_ICRFANTENNA_H
#define SIMPLA_ICRFANTENNA_H

#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/all.h"
#include "simpla/engine/all.h"
#include "simpla/physics/PhysicalConstants.h"

#include <cmath>

namespace simpla {
using namespace algebra;
using namespace data;

template <typename THost>
class ICRFAntenna {
    DOMAIN_POLICY_HEAD(ICRFAntenna);

   public:
    void Serialize(data::DataTable* res) const;
    void Deserialize(std::shared_ptr<DataTable> const& cfg);
    void Advance(Real time_now, Real dt);

    Field<host_type, Real, EDGE> J{m_host_, "name"_ = "J"};

    Vec3 m_amplify_{0, 0, 0};
    Real m_f_ = 1.0;
    Vec3 m_k_{0, 0, 0};
};

template <typename TM>
void ICRFAntenna<TM>::Serialize(data::DataTable* res) const {
    res->SetValue("Amplify", m_amplify_);
    res->SetValue("Frequency", m_f_);
    res->SetValue("WaveNumber", m_k_);
};
template <typename TM>
void ICRFAntenna<TM>::Deserialize(std::shared_ptr<DataTable> const& cfg) {
    m_amplify_ = cfg->GetValue<Vec3>("Amplify", m_amplify_);
    m_f_ = cfg->GetValue<Real>("Frequency", m_f_);
    m_k_ = cfg->GetValue<Vec3>("WaveNumber", m_k_);
}

template <typename TM>
void ICRFAntenna<TM>::Advance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST
    J = [&](point_type const& x) -> Vec3 {
        Vec3 res;
        res = m_amplify_ * std::sin(m_k_[0] * x[0] + m_k_[1] * x[1] + m_k_[2] * x[2] + TWOPI * m_f_ * time_now);
        return res;
    };
}

}  // namespace simpla;
#endif  // SIMPLA_ICRFANTENNA_H
