//
// Created by salmon on 16-6-29.
//

#ifndef SIMPLA_EM_SCENARIO_H
#define SIMPLA_EM_SCENARIO_H

#include "../../src/simulation/Context.h"

namespace simpla { namespace scenario
{

class EM : public simulation::Context
{
    EM() { }

    virtual ~EM() { }

    virtual void setup();
};
}}//namespace simpla{namespace  scenario{

#endif //SIMPLA_EM_SCENARIO_H
