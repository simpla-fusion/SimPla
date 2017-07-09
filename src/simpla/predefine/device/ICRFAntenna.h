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
using namespace engine;

template <typename TM>
class ICRFAntenna : public engine::Domain {
    SP_OBJECT_HEAD(ICRFAntenna<TM>, engine::Domain)

   public:
    DOMAIN_HEAD(ICRFAntenna, TM)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const& cfg) override;

    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real dt) override;
    void DoAdvance(Real time_now, Real dt) override;

    Field<TM, Real, EDGE> J{this, "name"_ = "J"};

    Vec3 m_amplify_{0, 0, 0};
    Real m_f_ = 1.0;
    Vec3 m_k_{0, 0, 0};
};

template <typename TM>
bool ICRFAntenna<TM>::is_registered = engine::Domain::RegisterCreator<ICRFAntenna<TM>>();

template <typename TM>
std::shared_ptr<data::DataTable> ICRFAntenna<TM>::Serialize() const {
    auto res = engine::Domain::Serialize();
    res->SetValue("Amplify", m_amplify_);
    res->SetValue("Frequency", m_f_);
    res->SetValue("WaveNumber", m_k_);

    return res;
};
template <typename TM>
void ICRFAntenna<TM>::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    DoInitialize();
    engine::Domain::Deserialize(cfg);
    m_amplify_ = cfg->GetValue<Vec3>("Amplify", m_amplify_);
    m_f_ = cfg->GetValue<Real>("Frequency", m_f_);
    m_k_ = cfg->GetValue<Vec3>("WaveNumber", m_k_);

    Click();
}

template <typename TM>
void ICRFAntenna<TM>::DoInitialCondition(Real time_now) {
    Domain::DoInitialCondition(time_now);
}
template <typename TM>
void ICRFAntenna<TM>::DoBoundaryCondition(Real time_now, Real dt) {
    Domain::DoBoundaryCondition(time_now, dt);
}
template <typename TM>
void ICRFAntenna<TM>::DoAdvance(Real time_now, Real dt) {
    Domain::DoAdvance(time_now, dt);

    DEFINE_PHYSICAL_CONST
    J = [&](point_type const& x) -> Vec3 {
        Vec3 res;
        res = m_amplify_ * std::sin(m_k_[0] * x[0] + m_k_[1] * x[1] + m_k_[2] * x[2] + TWOPI * m_f_ * time_now);
        return res;
    };
}

}  // namespace simpla;
#endif  // SIMPLA_ICRFANTENNA_H
