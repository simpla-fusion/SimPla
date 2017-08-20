//
// Created by salmon on 16-11-7.
//

#ifndef SIMPLA_TIMEINTEGRATOR_H
#define SIMPLA_TIMEINTEGRATOR_H

#include "simpla/SIMPLA_config.h"

#include <memory>

#include "Schedule.h"

namespace simpla {
namespace engine {
class Context;

struct TimeIntegrator : public Schedule {
    SP_OBJECT_HEAD(TimeIntegrator, Schedule);

   public:
    void Synchronize() override;
    void NextStep() override;
    bool Done() const override { return GetTimeNow() >= GetTimeEnd() || Schedule::Done(); }

    virtual Real Advance(Real time_dt);

    SP_OBJECT_PROPERTY(Real, TimeNow);
    SP_OBJECT_PROPERTY(Real, TimeEnd);
    SP_OBJECT_PROPERTY(Real, TimeStep);
    SP_OBJECT_PROPERTY(Real, CFL);
};

}  //{ namespace engine
}  // namespace simpla

#endif  // SIMPLA_TIMEINTEGRATOR_H
