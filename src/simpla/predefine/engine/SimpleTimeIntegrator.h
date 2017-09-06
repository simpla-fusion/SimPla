//
// Created by salmon on 17-9-5.
//

#ifndef SIMPLA_SIMPLETIMEINTEGRATOR_H
#define SIMPLA_SIMPLETIMEINTEGRATOR_H

#include "simpla/engine/TimeIntegrator.h"
namespace simpla {
class SimpleTimeIntegrator : public engine::TimeIntegrator {
    SP_OBJECT_HEAD(SimpleTimeIntegrator, engine::TimeIntegrator);

   public:
    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;
    void Synchronize() override;
    void Advance(Real time_now, Real time_dt) override;
};
}  // namespace simpla
#endif  // SIMPLA_SIMPLETIMEINTEGRATOR_H
