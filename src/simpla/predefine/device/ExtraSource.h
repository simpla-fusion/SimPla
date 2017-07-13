//
// Created by salmon on 17-5-30.
//

#ifndef SIMPLA_EXTRASOURCE_H
#define SIMPLA_EXTRASOURCE_H

#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/algebra.h"
#include "simpla/engine/engine.h"
#include "simpla/physics/PhysicalConstants.h"

namespace simpla {
using namespace algebra;
using namespace data;
using namespace engine;

template <typename TM>
class ExtraSource {
   public:
    DOMAIN_POLICY_HEAD(ExtraSource);

    void Serialize(data::DataTable* res) const;
    void Deserialize(std::shared_ptr<DataTable> const& cfg);

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real dt);
    void Advance(Real time_now, Real dt);

    std::shared_ptr<Attribute> m_attrr_;
};

template <typename TM>
void ExtraSource<TM>::Serialize(data::DataTable* res) const {};

template <typename TM>
void ExtraSource<TM>::Deserialize(std::shared_ptr<DataTable> const& cfg) {}

template <typename TM>
void ExtraSource<TM>::InitialCondition(Real time_now) {}
template <typename TM>
void ExtraSource<TM>::BoundaryCondition(Real time_now, Real dt) {}
template <typename TM>
void ExtraSource<TM>::Advance(Real time_now, Real dt) {}

}  // namespace simpla;
#endif  // SIMPLA_EXTRASOURCE_H
