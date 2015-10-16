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
#include "../gtl/macro.h"

#include "calculate/calculate.h"
#include "interpolate/interpolate.h"

namespace simpla
{

template<typename ...> class Manifold;


template<typename TGeo, typename ...Policies>
class Manifold<TGeo, Policies ...>
		: public TGeo, public Policies ...
{


	typedef TGeo geometry_type;

	typedef Manifold<geometry_type, Policies ...> this_type;

public:

	Manifold() : Policies(static_cast<geometry_type &>(*this))... { }

	virtual ~Manifold() { }

	Manifold(this_type const &other) : geometry_type(other), Policies(other)... { }

	this_type &operator=(const this_type &other)
	{
		this_type(other).swap(*this);
		return *this;
	}


private:

	TEMPLATE_DISPATCH_DEFAULT(load)

	TEMPLATE_DISPATCH(swap, inline,)

	TEMPLATE_DISPATCH(print, inline, const)

public:
	void swap(const this_type &other) { _dispatch_swap<geometry_type, Policies...>(other); }

	template<typename TDict>
	void load(TDict const &dict)
	{
		auto d = dict["Manifold"];
		_dispatch_load<geometry_type, Policies...>(d);
	}

	template<typename OS>
	OS &print(OS &os) const
	{
		os << "Manifold={" << std::endl;

		_dispatch_print<geometry_type, Policies...>(os);

		os << "}, # Manifold " << std::endl;
		return os;
	}


}; //class Manifold


}//namespace simpla

#endif /* CORE_MANIFOLD_H_ */
