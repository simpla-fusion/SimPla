/*
 * \file explicit_em.h
 *
 *  Created on: 2013年12月12日
 *      Author: salmon
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
class LuaObject;

struct Context
{

	Context(LuaObject const &);
	Context();
	~Context();

	void Load(LuaObject const &);

	std::function<void(std::string const &)> Dump;

	std::function<void(std::ostream &)> Print;

	std::function<void()> NextTimeStep;

	bool empty() const
	{
		return false;
	}
	operator bool() const
	{
		return empty();
	}

};

inline std::ostream & operator<<(std::ostream & os, Context const &self)
{
	self.Print(os);
	return os;
}

template<typename TC, typename TDict>
void CreateContext(TDict const &dict, Context* ctx)
{

	std::shared_ptr<TC> ctx_ptr(new TC);
	ctx_ptr->Load(dict);
	using namespace std::placeholders;
	ctx->Print = std::bind(&TC::template Print<std::ostream>, ctx_ptr, _1);
	ctx->Dump = std::bind(&TC::Dump, ctx_ptr, _1);
	ctx->NextTimeStep = std::bind(&TC::NextTimeStep, ctx_ptr);

	LOGGER << *ctx_ptr;

}
}
// namespace simpla

#endif /* EXPLICIT_EM_H_ */
