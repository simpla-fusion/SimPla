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
#include "primitives/properties.h"
#include "fetl/fetl.h"
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
typedef TR1::shared_ptr<Object> ObjectHolder;
inline void eval_(TR1::function<void(void)> & f)
{
	f();
}
class BaseContext
{
public:

	std::map<std::string, TR1::shared_ptr<Object> > objects;
	std::list<TR1::function<void()> > functions;
	typedef TR1::shared_ptr<BaseContext> Holder;

	BaseContext()
	{
	}
	virtual ~BaseContext()
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

	template<typename TG> virtual TG const & getGrid()=0;

	template<typename TOBJ> virtual TR1::shared_ptr<TOBJ> CreateObject()=0;

	template<typename TOBJ>
	TR1::shared_ptr<TOBJ> CreateObject(std::string const & name,
			TR1::shared_ptr<TOBJ> obj = TR1::shared_ptr<TOBJ>())
	{
		TR1::shared_ptr<TOBJ> res;

		if (objects.find(name) != objects.end())
		{
			ERROR << "Can not create new object! Object\"" << name
					<< "\" has been defined. ";
		}
		else if (obj != TR1::shared_ptr<TOBJ>())
		{
			res = obj;
		}
		else
		{
			res = CreateObject<TOBJ>();
		}

		if (name != "")
		{
			objects[name] = res;
		}
	}

	template<typename TOBJ>
	TR1::shared_ptr<TOBJ> CreateObject(std::string const & name, TOBJ* obj)
	{
		return CreateObject(name, TR1::shared_ptr<TOBJ>(obj));
	}

	template<typename TOBJ>
	TR1::shared_ptr<TOBJ> FindObject(std::string const & name)
	{
		TR1::shared_ptr<TOBJ> res;

		if (name != "")
		{
			std::map<std::string, typename Object::Holder>::iterator it =
					objects.find(name);
			if (it != objects.end() && it->second->CheckType(typeid(TOBJ)))
			{
				res = TR1::dynamic_pointer_cast<TOBJ>(it->second);
			}
		}

		return (res);
	}

	template<typename TOBJ>
	TR1::shared_ptr<TOBJ> GetObject(const std::string & name = "")
	{
		TR1::shared_ptr<TOBJ> res = FindObject<TOBJ>(name);

		if (res == TR1::shared_ptr<TOBJ>())
		{
			res = CreateObject<TOBJ>(name);
		}

		return (res);
	}
	void DeleteObject(std::string const & name)
	{
		objects.erase(name);
	}

	void Eval()
	{
		++counter_;
		std::for_each(functions.begin(), functions.end(), eval_);

	}
private:
	size_t counter_;
	Real timer_;
	std::list<std::string, TR1::shared_ptr<BaseContext> > neighbours_;

};

using namespace fetl;

template<typename TV, typename TG>
class Context: public BaseContext
{
public:

	typedef Context<TV, TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(TV,TG)

	Grid grid;

	Context(ptree const & pt)

	~Context()
	{
	}

	template<typename >
	Grid const & getGrid()
	{
		return grid;
	}

	TR1::shared_ptr<BaseContext> Create(ptree const &properties)
	{
		return TR1::dynamic_pointer_cast<BaseContext>(
				Holder(new Context(properties)));
	}

	template<typename TOBJ>
	inline TR1::shared_ptr<TOBJ> CreateObject()
	{
		return TR1::shared_ptr<TOBJ>(new TOBJ(grid));
	}

private:
	Context(Context const &);
	Context & operator=(Context const &);
};

}
// namespace simpla
#endif   // SRC_ENGINE_CONTEXT_H_
