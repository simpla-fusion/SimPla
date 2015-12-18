/*
 * \file explicit_em.h
 *
 * @date  2013-12-12
 *      @author  salmon
 */

#ifndef EXPLICIT_EM_H_
#define EXPLICIT_EM_H_

#include <algorithm>
#include <string>

#include "../../core/design_pattern/factory.h"
#include "../../core/flow_control/context_base.h"
#include "../../core/utilities/lua_state.h"

namespace simpla
{
/**
 *   \defgroup  Application Application
 *   @{
 *      \defgroup  FieldSolver field Solver
 *      \defgroup  ParticleEngine Particle Engine
 *   @}
 *
 */

typedef Factory<std::string, ContextBase, LuaObject> context_factory;

void RegisterEMContextCartesian(context_factory *);
void RegisterEMContextCylindrical(context_factory *);

context_factory RegisterContext()
{

	context_factory factory;

	RegisterEMContextCartesian(&factory);
	RegisterEMContextCylindrical(&factory);

	return std::move(factory);
}

}
// namespace simpla

#endif /* EXPLICIT_EM_H_ */
