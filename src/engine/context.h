/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$
 * Engine/Context.h
 *
 *  Created on: 2010-11-10
 *      Author: salmon
 */

#ifndef SRC_ENGINE_CONTEXT_H_
#define SRC_ENGINE_CONTEXT_H_
#include <list>
#include <string>
#include <map>
#include <typeinfo>
#include "include/simpla_defs.h"

namespace simpla
{

/** Context
 * NOTE
 *
 *  XXX::registerFunction(ctx,...) parse the configure, setup  parameters
 *
 *  ctx->pre_process() 	create data memory  , set initial value
 *                      or restore data from files
 *                      prepare the compute environment  ;
 *
 *  ctx->process()      repeat the loop of process and output diagnosis data
 *
 *  ctx->post_process() destruct data  or dumping to file
 *
 * */

class Object;
inline void eval_(TR1::function<void(void)> & f)
{
	f();
}
class Context
{

public:
	typedef TR1::shared_ptr<Object> ObjectHolder;
	typedef TR1::shared_ptr<Context> Holder;

	TR1::shared_ptr<const Object> grid;

	std::map<std::string, TR1::shared_ptr<Object> > objects;
	std::list<TR1::function<void()> > functions;

	Context() :
			counter_(0), timer_(0)
	{
	}
	~Context()
	{
	}

	inline size_t Counter() const
	{
		return (counter_);
	}

	inline Real Time() const
	{
		return (timer_);
	}

	template<typename TG>
	void setGrid(TR1::shared_ptr<TG> const& pg)
	{
		grid = TR1::dynamic_pointer_cast<const Object>(pg);
	}

	template<typename TG>
	TG const & getGrid() const
	{
		if (grid->CheckType(typeid(TG)))
		{
			ERROR << "Grid type is not " << typeid(TG).name();
		}
		return (*TR1::dynamic_pointer_cast<const TG>(grid));
	}

	template<typename TO>
	void AddObject(std::string const & name, TR1::shared_ptr<TO> obj)
	{
		if (name != "")

		{
			std::map<std::string, typename Object::Holder>::iterator it =
					objects.find(name);

			if (it == objects.end())
			{
				objects[name] = obj;
			}
			else
			{
				ERROR << "Object\"" << name << "\" has been registered";
			}
		}
	}

	template<typename TO>
	void AddObject(std::string const & name, TO* obj)
	{
		AddObject(name, TR1::shared_ptr<TO>(obj));

	}

	template<typename TO>
	TR1::shared_ptr<TO> CreateObject(std::string const & name)
	{
		TR1::shared_ptr<TO> res(new TO(getGrid<typename TO::Grid>()));
		AddObject(name, res);
		return (res);
	}

	template<typename TO> inline TR1::shared_ptr<TO> FindObject(
			std::string const & name)
	{
		TR1::shared_ptr<TO> res;
		std::map<std::string, typename Object::Holder>::iterator it =
				objects.find(name);
		if (it != objects.end() && it->second->CheckType(typeid(TO)))
		{
			res = TR1::dynamic_pointer_cast<TO>(it->second);
		}
		return (res);
	}

	template<typename TF>
	TR1::shared_ptr<TF> GetObject(const std::string & name)
	{
		TR1::shared_ptr<TF> res;

		if (name != "")
		{
			res = FindObject<TF>(name);
		}

		if (res == TR1::shared_ptr<TF>())
		{
			res = CreateObject<TF>(name);

		}

		return (res);
	}
	void DeleteObject(std::string const & name)
	{
		objects.erase(name);
	}

	inline void Eval()
	{
		++counter_;
		std::for_each(functions.begin(), functions.end(), eval_);

	}

private:
	Context(Context const &);
	Context & operator=(Context const &);

	std::map<std::string, TR1::shared_ptr<Object> > neighbours_;

	size_t counter_;
	Real timer_;

};
} // namespace simpla
#endif   // SRC_ENGINE_CONTEXT_H_
