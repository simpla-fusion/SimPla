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

	std::list<TR1::shared_ptr<Object> > opool_;

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
	TR1::shared_ptr<TOBJ> GetObject(std::string const & name = "");

	void Eval();

private:
	size_t counter_;
	Real timer_;
	std::list<TR1::shared_ptr<BaseContext> > neighbours_;

	virtual BaseGrid const * getGridPtr() const =0;

}
;

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

}
// namespace simpla

#endif /* DOMAIN_H_ */
