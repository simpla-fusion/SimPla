/*
 * \file explicit_em.h
 *
 * \date  2013年12月12日
 *      \author  salmon
 */

#ifndef EXPLICIT_EM_H_
#define EXPLICIT_EM_H_

#include <functional>
#include <iostream>
#include <memory>
#include <string>

#include "../../src/utilities/log.h"
#include "../../src/utilities/lua_state.h"

namespace simpla
{
/**
 *   @defgroup Application
 *   @{
 *      @defgroup FieldSolver
 *      @defgroup ParticleEngine
 *   @}
 *
 */

/**
 *  @brief Context wrapper
 */
struct Context
{
	Context(LuaObject const & dict);
	Context();
	~Context();

	void Load(LuaObject const & dict);

	std::function<std::string(std::string const &, bool)> Save;

	std::function<void()> NextTimeStep;

	std::function<std::string()> Begin;
	std::function<std::string()> End;

	bool empty() const
	{
		return false;
	}
	operator bool() const
	{
		return empty();
	}

};

}
// namespace simpla

#endif /* EXPLICIT_EM_H_ */
