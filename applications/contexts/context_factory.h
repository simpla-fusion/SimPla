/*
 * \file explicit_em.h
 *
 * \date  2013-12-12
 *      \author  salmon
 */

#ifndef EXPLICIT_EM_H_
#define EXPLICIT_EM_H_

#include <algorithm>
#include <string>

#include "../../src/flow_control/context_base.h"
#include "../../src/utilities/factory.h"
#include "../../src/utilities/lua_state.h"

namespace simpla
{
/**
 *   \defgroup  Application Application
 *   @{
 *      \defgroup  FieldSolver Field Solver
 *      \defgroup  ParticleEngine Particle Engine
 *   @}
 *
 */

typedef Factory<std::string, ContextBase, LuaObject> context_factory;

void RegisterEMContextCartesian(context_factory *);
//void RegisterEMContextCylindical(context_factory *);

context_factory RegisterContext()
{

	context_factory factory;

	RegisterEMContextCartesian(&factory);
//	RegisterEMContextCylindical(&factory);

	return std::move(factory);
}

}
// namespace simpla

#endif /* EXPLICIT_EM_H_ */
