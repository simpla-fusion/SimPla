/*
 * BaseContext.h
 *
 *  Created on: 2012-10-26
 *      Author: salmon
 */

#ifndef BASECONTEXT_H_
#define BASECONTEXT_H_

#include "include/simpla_defs.h"
#include "object.h"
#include "compound.h"
#include <list>
#include <string>
#include <map>
#include <typeinfo>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>

#include "physics/physical_constants.h"
#include "utilities/properties.h"

namespace simpla
{

class BaseContext
{
public:
	//FIXME Need garbage collection of objects!!

	std::map<std::string, TR1::function<TR1::shared_ptr<Object>(ptree const&)> > objFactory_;

	std::map<std::string, TR1::function<TR1::function<void(void)>(ptree const&)> > moduleFactory_;

	ptree env;

	TR1::shared_ptr<CompoundObject> objects;

	PhysicalConstants PHYS_CONSTANTS;

	BaseContext();

	virtual void Parse(ptree const&pt);

	virtual ~BaseContext();

	virtual std::string Summary() const=0;

	virtual void InitLoad(ptree const&pt)
	{
		objects = CompoundObject::Create(this, pt);
	}

	virtual void Process(ptree const&pt);

	inline size_t Counter() const
	{
		return (counter_);
	}

	inline Real Timer() const
	{
		return (timer_);
	}

	inline void PushClock()
	{
		timer_ += dt;
		++counter_;
	}

	template<typename T>
	inline boost::optional<T> GetEnv(std::string const &name,
			boost::optional<ptree const &> pt) const
	{
		boost::optional<T> res(false, T());
		if (!!pt)
		{
			if (boost::optional<const ptree &> apt = pt->get_child_optional(
					name))
			{
				if (apt->data().substr(0, 5) != "$ENV{")
				{
					res = apt->get_value_optional<T>();
				}
				else
				{
					res = env.get_optional<T>(
							apt->data().substr(5, apt->data().size() - 6));
				}
			}
		}
		return res;
	}

private:
	Real dt;
	size_t counter_;
	Real timer_;
	std::list<TR1::shared_ptr<BaseContext> > neighbours_;
}
;

}  // namespace simpla
#endif /* BASECONTEXT_H_ */
