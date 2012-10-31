/*
 * flow_control.h
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#ifndef FLOW_CONTROL_H_
#define FLOW_CONTROL_H_
#include "include/simpla_defs.h"
#include "engine/context.h"
#include "engine/modules.h"
#include "utilities/properties.h"

namespace simpla
{

namespace flow_control
{

class Clock: public Module
{
public:
	typedef Clock ThisType;

	BaseContext & ctx;

	Clock(BaseContext & d, const ptree & pt) :
			ctx(d)
	{
	}

	virtual ~Clock()
	{
	}
	static TR1::function<void(void)> Create(BaseContext* d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}

	virtual void Eval()
	{
		ctx.PushClock();

		LOG << "Counter: " << ctx.Counter() << "  Time: " << ctx.Timer();
	}

}
;


inline void eval_(TR1::function<void(void)> & f)
{
	f();
}
class Loop: public Module
{
public:
	typedef Loop ThisType;

	size_t maxstep;

	BaseContext & ctx;

	std::list<TR1::function<void(void)> > modules;

	Loop(BaseContext & d, const ptree & pt) :
			ctx(d), maxstep(pt.get("<xmlattr>.Steps", 1))
	{
		BOOST_FOREACH(const typename ptree::value_type &v, pt)
		{
			CHECK(v.first);
			if (v.first == "<xmlcomment>" || v.first == "<xmlattr>")
			{
				continue;
			}
			if (ctx.moduleFactory_.find(v.first) != ctx.moduleFactory_.end())
			{
				modules.push_back(ctx.moduleFactory_[v.first](v.second));
				LOG << "Add module " << v.first << " successed!";
			}
			else
			{
				WARNING << "Module type " << v.first << " is not registered!";
			}

		}
	}

	virtual ~Loop()
	{
	}
	static TR1::function<void(void)> Create(BaseContext* d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}

	virtual void Eval()
	{

		for (size_t i = 0; i < maxstep; ++i)
		{
			std::for_each(modules.begin(), modules.end(), eval_);
		}
	}

}
;

}  // namespace flow_control

}  // namespace simpla

#endif /* FLOW_CONTROL_H_ */
