//
// Created by salmon on 16-11-7.
//

#ifndef SIMPLA_TIMEINTEGRATOR_H
#define SIMPLA_TIMEINTEGRATOR_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Configurable.h>
#include <memory>
#include "SPObject.h"
namespace simpla {
namespace engine {
class Manager;
class TimeIntegrator : public concept::Configurable {
    SP_OBJECT_BASE(engine::TimeIntegrator);

   public:
    TimeIntegrator(Manager* manager = nullptr, std::shared_ptr<data::DataEntity> const& t = nullptr);
    virtual ~TimeIntegrator();
    virtual Real Advance(Real dt, int level = 0);
    virtual size_type NextTimeStep(Real dt);
    virtual size_type step() const;
    virtual bool remainingSteps() const;
    virtual Real timeNow() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  //{ namespace engine
}  // namespace simpla

#endif  // SIMPLA_TIMEINTEGRATOR_H
