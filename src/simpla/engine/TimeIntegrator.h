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
    virtual void CheckPoint() const;


    virtual void InitialCondition(Real time_now);
    virtual void BoundaryCondition(Real time_now, Real dt);
    virtual void ComputeFluxes(Real time_now, Real time_dt);
    virtual Real ComputeStableDtOnPatch(Real time_now, Real time_dt);
    virtual Real Advance(Real time_now, Real dt);

    virtual void Synchronize();
    virtual void NextStep();
    virtual void Run();
    virtual bool Done() const;

    SP_OBJECT_PROPERTY(size_type, Step);
    SP_OBJECT_PROPERTY(size_type, MaxStep);
    SP_OBJECT_PROPERTY(size_type, CheckPointInterval);
    SP_OBJECT_PROPERTY(size_type, DumpInterval);

    SP_OBJECT_PROPERTY(Real, TimeNow);
    SP_OBJECT_PROPERTY(Real, TimeEnd);
    SP_OBJECT_PROPERTY(Real, TimeStep);
    SP_OBJECT_PROPERTY(Real, CFL);
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_TIMEINTEGRATOR_H
