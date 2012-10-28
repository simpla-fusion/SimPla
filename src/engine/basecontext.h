/*
 * BaseContext.h
 *
 *  Created on: 2012-10-26
 *      Author: salmon
 */

#ifndef BASECONTEXT_H_
#define BASECONTEXT_H_

#include "include/simpla_defs.h"
#include <list>
#include <string>
#include <map>
#include <typeinfo>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>
#include "physics/physical_constants.h"
#include "object.h"

namespace simpla
{
class BaseGrid;

class BaseContext
{

public:

	Real dt;

	std::list<TR1::shared_ptr<Object> > opool_;

	std::map<std::string, TR1::shared_ptr<Object> > objects;

	std::list<TR1::function<void()> > modules;

	PhysicalConstants PHYS_CONSTANTS;

	BaseContext();

	virtual ~BaseContext();

	template<typename PT>
	BaseContext(PT const&pt) :
			dt(pt.get("dt", 1.0d)),

			PHYS_CONSTANTS(pt.get_child("PhysConstants")),

			counter_(0), timer_(0)
	{
	}

	virtual std::string Summary() const=0;

	inline size_t Counter() const
	{
		return (counter_);
	}

	inline Real Timer() const
	{
		return (timer_);
	}

	boost::optional<TR1::shared_ptr<Object> > FindObject(
			std::string const & name,
			std::type_info const &tinfo = typeid(void));

	boost::optional<TR1::shared_ptr<const Object> > FindObject(
			std::string const & name,
			std::type_info const &tinfo = typeid(void)) const;

	void DeleteObject(std::string const & name);

	template<typename TOBJ>
	TR1::shared_ptr<TOBJ> CreateObject();

	template<typename TOBJ>
	TR1::shared_ptr<TOBJ> GetObject(std::string const & name = "");

	void Eval();

	template<typename TG> inline TG const & Grid() const
	{
		return *reinterpret_cast<TG const *>(getGridPtr());
	}

private:
	size_t counter_;
	Real timer_;
	std::list<TR1::shared_ptr<BaseContext> > neighbours_;

	virtual BaseGrid const * getGridPtr() const =0;

}
;

template<typename TOBJ>
TR1::shared_ptr<TOBJ> BaseContext::CreateObject()
{
	return TR1::shared_ptr<TOBJ>(
			new TOBJ(*static_cast<typename TOBJ::Grid const *>(getGridPtr())));
}

template<typename TOBJ>
TR1::shared_ptr<TOBJ> BaseContext::GetObject(std::string const & name)
{

	if (name != "")
	{
		std::map<std::string, Object::Holder>::iterator it = objects.find(name);
		if (it != objects.end())
		{
			if (it->second->CheckType(typeid(TOBJ)))
			{
				return TR1::dynamic_pointer_cast<TOBJ>(it->second);
			}
			else
			{
				ERROR << "Object " << name << "can not been created as "
						<< typeid(TOBJ).name();
			}
		}
	}
	TR1::shared_ptr<TOBJ> res = CreateObject<TOBJ>();

	if (name != "")
	{
		objects[name] = TR1::dynamic_pointer_cast<Object>(res);
	}

	return res;
}

}  // namespace simpla

#endif /* BASECONTEXT_H_ */
