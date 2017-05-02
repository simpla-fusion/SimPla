//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_PEC_H
#define SIMPLA_PEC_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include <simpla/physics/PhysicalConstants.h>
#include <simpla/utilities/Log.h>
namespace simpla {
using namespace engine;

/**
 *  @ingroup
 *  @brief   PEC
 */
template <typename TM>
class PEC : public engine::Domain {
    SP_OBJECT_HEAD(PEC<TM>, engine::Domain)
    typedef TM mesh_type;

    DOMAIN_HEAD(PEC, engine::Domain)

    void InitialCondition(Real time_now) override;
    void Advance(Real time, Real dt) override;

    field_type<EDGE> E{m_mesh_, "name"_ = "E"};
    field_type<FACE> B{m_mesh_, "name"_ = "B"};
};
template <typename TM>
REGISTER_CREATOR(PEC<TM>)

template <typename TM>
void PEC<TM>::InitialCondition(Real time_now) {}

template <typename TM>
void PEC<TM>::Advance(Real time, Real dt) {
    E = 0.0;
    B = 0.0;
}

}  // namespace simpla

#endif  // SIMPLA_PEC_H
