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

template <typename THost>
class ICRFAntenna {
    SP_ENGINE_POLICY_HEAD(ICRFAntenna);

   public:
    void Advance(Real time_now, Real dt);
    void InitialCondition(Real time_now);
    //    void TagRefinementCells(Real time_now);

    Field<host_type, Real, EDGE> J{m_host_, "name"_ = "J"};

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
void ICRFAntenna<TM>::InitialCondition(Real time_now) {
    m_host_->GetMesh()->SetEmbeddedBoundary(m_host_->GetName(), m_host_->GetGeoBody());
}
template <typename TM>
void ICRFAntenna<TM>::Deserialize(std::shared_ptr<const data::DataNode> cfg) {
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
