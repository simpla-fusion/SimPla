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
    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void Synchronize() override;
    Real Advance(Real time_now, Real time_dt) override;
    bool Done() const override;

    void CheckPoint() const override;
    void Dump() const override;
};
}  // namespace simpla
#endif  // SIMPLA_SIMPLETIMEINTEGRATOR_H
