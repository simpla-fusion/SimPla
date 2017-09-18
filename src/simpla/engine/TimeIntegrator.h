//
// Created by salmon on 17-9-5.
//

#ifndef SIMPLA_TIMEINTEGRATOR_H
#define SIMPLA_TIMEINTEGRATOR_H

#include "Scenario.h"
namespace simpla {
namespace engine {
class TimeIntegrator : public Scenario {
    SP_OBJECT_HEAD(TimeIntegrator, Scenario);

   public:
    virtual void InitialCondition(Real time_now);
    virtual void BoundaryCondition(Real time_now, Real dt);
    virtual void ComputeFluxes(Real time_now, Real time_dt);
    virtual Real ComputeStableDtOnPatch(Real time_now, Real time_dt);
    virtual void Advance(Real time_now, Real dt);

    void DoSetUp() override;
    void DoTearDown() override;

    void Synchronize(int level) override;
    void NextStep() override;
    void Run() override;
    bool Done() const override;

    SP_OBJECT_PROPERTY(size_type, CheckPointInterval);
    SP_OBJECT_PROPERTY(size_type, DumpInterval);
    SP_OBJECT_PROPERTY(size_type, MaxStep);
    SP_OBJECT_PROPERTY(Real, CFL);

    Real GetTimeNow() const override;
    void SetTimeNow(Real);
    Real GetTimeEnd() const;
    void SetTimeEnd(Real);
    Real GetTimeStep() const;
    void SetTimeStep(Real);
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_TIMEINTEGRATOR_H
