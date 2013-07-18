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

#include "primitives/primitives.h"
#include "physics/physical_constants.h"

namespace simpla
{

class BaseContext
{
public:
	//FIXME Need garbage collection of objects!!

	std::map<std::string, TR1::function<TR1::shared_ptr<Object>(ptree const&)> > objFactory_;

	std::map<std::string, TR1::function<TR1::function<void(void)>(ptree const&)> > moduleFactory_;

	TR1::shared_ptr<CompoundObject> objects;

	PhysicalConstants PHYS_CONSTANTS;

	BaseContext();


	virtual ~BaseContext();


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



private:
	Real dt;
	size_t counter_;
	Real timer_;
	std::list<TR1::shared_ptr<BaseContext> > neighbours_;
}
;

}  // namespace simpla
#endif /* BASECONTEXT_H_ */
