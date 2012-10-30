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
class BaseGrid;

class BaseContext
{

public:

	std::map<std::string, TR1::shared_ptr<Object> > objects;

	std::map<std::string, TR1::function<TR1::shared_ptr<Object>(void)> > objFactory_;

	std::list<TR1::function<void(void)> > modules;

	std::map<std::string, TR1::function<TR1::function<void(void)>(ptree const&)> > moduleFactory_;

	PhysicalConstants PHYS_CONSTANTS;

	BaseContext();
	BaseContext(ptree const&pt);
	virtual ~BaseContext();

	virtual std::string Summary() const=0;

	inline size_t Counter() const
	{
		return (counter_);
	}

	inline Real Timer() const
	{
		return (timer_);
	}
	void Load(ptree const & pt);

	void Eval(size_t maxstep);

	boost::optional<TR1::shared_ptr<Object> > FindObject(
			std::string const & name,
			std::type_info const &tinfo = typeid(void));

	boost::optional<TR1::shared_ptr<const Object> > FindObject(
			std::string const & name,
			std::type_info const &tinfo = typeid(void)) const;

	void DeleteObject(std::string const & name);




private:
	Real dt;
	size_t counter_;
	Real timer_;
	std::list<TR1::shared_ptr<BaseContext> > neighbours_;
}
;

}  // namespace simpla

#endif /* BASECONTEXT_H_ */
