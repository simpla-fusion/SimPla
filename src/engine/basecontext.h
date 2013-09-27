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

	std::map<std::string, std::function<std::shared_ptr<Object>(PTree const&)> > objFactory_;

	std::map<std::string, std::function<std::function<void(void)>(PTree const&)> > moduleFactory_;

	CompoundObject objects;

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
	std::list<std::shared_ptr<BaseContext> > neighbours_;
}
;

}  // namespace simpla
#endif /* BASECONTEXT_H_ */
