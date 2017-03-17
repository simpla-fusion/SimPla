//
// Created by salmon on 16-11-7.
//

#ifndef SIMPLA_TIMEINTEGRATOR_H
#define SIMPLA_TIMEINTEGRATOR_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include "DomainView.h"
#include "Manager.h"

namespace simpla {
namespace engine {
class TimeIntegrator {
    Manager* m_manger_;

   public:
    TimeIntegrator() {}
    virtual ~TimeIntegrator() {}

    virtual Real Advance(Real dt, int level = 0);
    virtual size_type NextTimeStep(Real dt) { return 0; };
    virtual size_type step() const { return 0; };
    virtual bool remainingSteps() const { return 0; };
    virtual Real timeNow() const { return 0.0; }
};
}
}  // namespace simpla { namespace simulation

#endif  // SIMPLA_TIMEINTEGRATOR_H
