/*
 * Domain.h
 *
 *  Created on: 2012-10-9
 *      Author: salmon
 */

#ifndef DOMAIN_H_
#define DOMAIN_H_

#include <list>
#include <string>
#include <map>
#include <typeinfo>
#include <boost/foreach.hpp>

#include "include/simpla_defs.h"
#include "physics/physical_constants.h"
#include "object.h"

namespace simpla
{

class BaseGrid;

class BaseContext
{

public:

	const Real dt;

	std::list<TR1::shared_ptr<Object> > unnamed_objects;

	std::map<std::string, TR1::shared_ptr<Object> > objects;

	std::list<TR1::function<void()> > modules;

	PhysicalConstants PHYS_CONSTANTS;

	BaseContext(ptree const&pt);

	virtual ~BaseContext();

	virtual std::string Summary() const=0;

	inline size_t Counter() const
	{
		return (counter_);
	}

	inline Real Time() const
	{
		return (timer_);
	}

	boost::optional<Object &> FindObject(std::string const & name,
			std::type_info const &tinfo);

	void DeleteObject(std::string const & name);

	template<typename TOBJ>
	TOBJ & GetObject(std::string const & name = "");

	void Eval();

private:
	size_t counter_;
	Real timer_;
	std::list<TR1::shared_ptr<BaseContext> > neighbours_;

	virtual BaseGrid const * getGridPtr() const =0;

}
;

template<typename TOBJ>
TOBJ & BaseContext::GetObject(std::string const & name)
{

	if (name != "")
	{
		std::map<std::string, Object::Holder>::iterator it = objects.find(name);
		if (it != objects.end())
		{
			if (it->second->CheckType(typeid(TOBJ)))
			{
				return *TR1::dynamic_pointer_cast<TOBJ>(it->second);
			}
			else
			{
				ERROR << "Object " << name << "can not been created as "
						<< typeid(TOBJ).name();
			}
		}
	}

	TR1::shared_ptr<TOBJ> res(
			new TOBJ(*static_cast<typename TOBJ::Grid const *>(getGridPtr())));
	if (name != "")
	{
		objects[name] = res;
	}
	return *res;
}

template<typename TG>
class Context: public BaseContext
{
public:
	typedef Context ThisType;

	typedef TG Grid;

	typedef TR1::shared_ptr<ThisType> Holder;

	Grid grid;

	Context(const ptree & pt) :
			BaseContext(pt), grid(pt.get_child("Grid"))
	{
		LoadModules(pt);
	}

	~Context()
	{
		// Check ref count of objects
	}

	static TR1::shared_ptr<ThisType> Create(ptree const & pt);

	void LoadModules(ptree const & pt);

	virtual std::string Summary() const;

private:
	Grid const * getGridPtr() const
	{
		return &grid;
	}
	Context(ThisType const &);
	Context & operator=(ThisType const &);

}
;

template<typename TG>
TR1::shared_ptr<Context<TG> > Context<TG>::Create(ptree const & pt)
{
	return TR1::shared_ptr<Context<TG> >(new ThisType(pt));
}
template<typename TG>
inline std::string Context<TG>::Summary() const
{
	std::ostringstream os;

	os

	<< PHYS_CONSTANTS.Summary()

	<< SINGLELINE << std::endl

	<< std::setw(20) << "dt : " << dt << std::endl

	<< grid.Summary() << std::endl

	<< SINGLELINE << std::endl;

	return os.str();

}

}
// namespace simpla

#endif /* DOMAIN_H_ */
