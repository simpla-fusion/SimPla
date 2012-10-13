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
#include "include/simpla_defs.h"
#include "utilities/properties.h"
#include "fetl/fetl.h"
#include "fetl/grid/grid.h"

namespace simpla
{

class Object;

class Domain
{
public:
	typedef Domain ThisType;

	std::map<std::string, TR1::shared_ptr<Object> > objects;

	std::list<TR1::function<void()> > functions;

	typedef TR1::shared_ptr<ThisType> Holder;

	const Real dt;

	ptree PHYS_CONSTANTS;

	Domain(const ptree & pt) :

			PHYS_CONSTANTS(pt.get_child("PhysConstants")),

			dt(pt.get<Real>("dt")),

			counter_(0),

			timer_(0)
	{
	}

	~Domain()
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
	TG const & grid()
	{
		return *TR1::dynamic_pointer_cast<const TG>(grid_);
	}

	template<typename TOBJ>
	inline TR1::shared_ptr<TOBJ> CreateObject(std::string const& name = "")
	{

		if (name != "" && objects.find(name) != objects.end())
		{
			ERROR << "Can not create new object! Object\"" << name
					<< "\" has been defined. ";
		}

		TR1::shared_ptr<TOBJ> res(new TOBJ(grid<typename TOBJ::Grid>()));

		if (name != "")
		{
			objects[name] = res;
		}

		return res;

	}
	template<typename TOBJ>
	inline TR1::shared_ptr<TOBJ> AddObject(std::string const& name,
			TR1::shared_ptr<TOBJ> o)
	{

		if (name != "" && objects.find(name) != objects.end())
		{
			ERROR << "Can not create new object! Object\"" << name
					<< "\" has been defined. ";
		}

		objects[name] = TR1::dynamic_pointer_cast<Object>(o);

		return o;

	}

	template<typename TOBJ>
	inline TR1::shared_ptr<TOBJ> AddObject(std::string const& name, TOBJ * o)
	{
		return AddObject(name, TR1::shared_ptr<TOBJ>(o));
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
	TR1::shared_ptr<TOBJ> GetObjectPtr(const std::string & name)
	{
		TR1::shared_ptr<TOBJ> res = FindObject<TOBJ>(name);

		if (res == TR1::shared_ptr<TOBJ>() && name != "")
		{
			res = CreateObject<TOBJ>(name);
		}

		return (res);
	}

	template<typename TOBJ>
	TOBJ & GetObject(std::string const & name = "")
	{
		return *GetObjectPtr<TOBJ>(name);
	}

	void DeleteObject(std::string const & name)
	{
		objects.erase(name);
	}
	static inline void eval_(TR1::function<void(void)> & f)
	{
		f();
	}
	void Eval()
	{
		++counter_;
		std::for_each(functions.begin(), functions.end(), eval_);
	}
private:

	Domain(ThisType const &);
	Domain & operator=(ThisType const &);
	size_t counter_;
	Real timer_;
	TR1::shared_ptr<BaseGrid> grid_;
	std::list<TR1::shared_ptr<ThisType> > neighbours_;
}
;

}
// namespace simpla

#endif /* DOMAIN_H_ */
