//
// Created by salmon on 17-6-5.
//

#ifndef SIMPLA_ICRFANTENNA_H
#define SIMPLA_ICRFANTENNA_H

#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/Algebra.h"
#include "simpla/engine/Engine.h"
#include "simpla/physics/PhysicalConstants.h"

#include <cmath>

namespace simpla {

using namespace data;

template <typename THost>
class ICRFAntenna {
    SP_ENGINE_POLICY_HEAD(ICRFAntenna);

   public:
    void Serialize(data::DataTable* res) const;
    void Deserialize(std::shared_ptr<DataTable> const& cfg);
    void Advance(Real time_now, Real dt);
    void InitialCondition(Real time_now);
    void TagRefinementCells(Real time_now);

    Field<host_type, Real, EDGE> J{m_host_, "name"_ = "J"};

    Vec3 m_amplify_{0, 0, 0};
    Real m_f_ = 1.0;
    Vec3 m_k_{0, 0, 0};
};
template <typename TM>
void ICRFAntenna<TM>::TagRefinementCells(Real time_now) {
    if (m_host_->GetMesh()->GetBlock()->GetLevel() > 0) { return; }
    m_host_->GetMesh()->TagRefinementCells(m_host_->GetMesh()->GetRange(m_host_->GetName() + "_BOUNDARY_3"));
}
template <typename TM>
void ICRFAntenna<TM>::Serialize(data::DataTable* res) const {
    res->SetValue("Amplify", m_amplify_);
    res->SetValue("Frequency", m_f_);
    res->SetValue("WaveNumber", m_k_);
};

template <typename TM>
void ICRFAntenna<TM>::InitialCondition(Real time_now) {}
template <typename TM>
void ICRFAntenna<TM>::Deserialize(std::shared_ptr<DataTable> const& cfg) {
    m_amplify_ = cfg->GetValue<Vec3>("Amplify", m_amplify_);
    m_f_ = cfg->GetValue<Real>("Frequency", m_f_);
    m_k_ = cfg->GetValue<Vec3>("WaveNumber", m_k_);
}

template <typename TM>
void ICRFAntenna<TM>::Advance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST

    SP_CMD((J = [=](point_type const& x) -> nTuple<Real, 3> {
        nTuple<Real, 3> res =
            m_amplify_ * std::sin(m_k_[0] * x[0] + m_k_[1] * x[1] + m_k_[2] * x[2] + TWOPI * m_f_ * time_now);
        return res;
    }));
}

}  // namespace simpla;
#endif  // SIMPLA_ICRFANTENNA_H
