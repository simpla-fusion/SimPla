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
#include <string>
#include <memory>
namespace simpla
{
class LuaObject;

struct Context
{

	Context(LuaObject const &);
	Context();
	~Context();

	void Load(LuaObject const &);

	std::function<void(std::string const &)> DumpData;

	std::function<void(std::ostream &)> Save;

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
	self.Save(os);
	return os;
}

template<typename TC, typename TDict>
void CreateContext(TDict const &dict, Context* ctx)
{

	std::shared_ptr<TC> ctx_ptr(new TC);
	ctx_ptr->Load(dict);
	using namespace std::placeholders;
	ctx->Save = std::bind(&TC::Save, ctx_ptr, _1);
	ctx->DumpData = std::bind(&TC::DumpData, ctx_ptr, _1);
	ctx->NextTimeStep = std::bind(&TC::NextTimeStep, ctx_ptr);

}
}
// namespace simpla

#endif /* EXPLICIT_EM_H_ */
