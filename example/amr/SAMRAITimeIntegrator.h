//
// Created by salmon on 16-11-8.
//

#ifndef SIMPLA_SAMRAITIMEINTEGRATOR_H
#define SIMPLA_SAMRAITIMEINTEGRATOR_H

#include <memory>
#include <string>
#include <simpla/simulation/TimeIntegrator.h>

namespace simpla
{

std::shared_ptr<simulation::TimeIntegrator>
create_samrai_time_integrator(std::string const &name,
                              std::shared_ptr<mesh::Worker> const &w = nullptr);


}
#endif //SIMPLA_SAMRAITIMEINTEGRATOR_H
