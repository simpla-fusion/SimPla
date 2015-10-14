/**
 * @file manifold.h
 *
 * @date 2015-2-9
 * @author salmon
 */

#ifndef CORE_MANIFOLD_H_
#define CORE_MANIFOLD_H_

#include <iostream>
#include <memory>
#include "calculate/calculate.h"
#include "interpolate/interpolate.h"

namespace simpla
{

template<typename ...> class Manifold;


template<typename TGeo, typename ...Polices>
class Manifold<TGeo, Polices ...>
		: public TGeo, public Polices ...
{
public:

	typedef Manifold<TGeo, Polices ...> this_type;

	typedef TGeo geometry_type;

	Manifold() : Polices(static_cast<geometry_type &>(*this))... { }

	virtual ~Manifold() { }

	Manifold(this_type const &other) : geometry_type(other) { }

	void swap(const this_type &other) { geometry_type::swap(other); }

	geometry_type const &geometry() const { return static_cast<geometry_type &>(*this); }

	this_type &operator=(const this_type &other)
	{
		this_type(other).swap(*this);
		return *this;
	}


	template<typename OS>
	OS &print(OS &os) const
	{
		os << "Manifold<>" << std::endl;
		return os;
	}

	static std::string get_type_as_string() { return "Manifold< >"; }

}; //class Manifold


}//namespace simpla

#endif /* CORE_MANIFOLD_H_ */
