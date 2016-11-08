//
// Created by salmon on 16-11-8.
//

#ifndef SIMPLA_SAMRAITIMEINTEGRATOR_H
#define SIMPLA_SAMRAITIMEINTEGRATOR_H

#include <simpla/simulation/TimeIntegrator.h>
#include <memory>
#include <string>

namespace simpla
{

std::shared_ptr<simulation::TimeIntegrator> create_samrai_time_integrator(std::string const &name);

}
#endif //SIMPLA_SAMRAITIMEINTEGRATOR_H
