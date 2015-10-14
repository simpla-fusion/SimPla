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

template<typename ...> class Domain;

template<typename ...> class Field;

template<typename ...> class Manifold;


template<typename TGeo, typename CalculateTag, typename InterpolatorTag, typename ...Others>
class Manifold<TGeo, CalculateTag, InterpolatorTag, Others ...>
		:

				public TGeo,

				public calculate::Calculate<TGeo, CalculateTag>,

				public interpolate::Interpolate<TGeo, InterpolatorTag>,

				public Others ...
{
public:

	typedef Manifold<TGeo, CalculateTag, InterpolatorTag, Others ...> this_type;

	typedef TGeo geometry_type;

	typedef calculate::Calculate <TGeo, CalculateTag> calculate_policy;

	typedef interpolate::Interpolate <TGeo, InterpolatorTag> interpolator_policy;

	Manifold()
	{
	}

	virtual ~Manifold()
	{
	}

	Manifold(this_type const &other) :
			geometry_type(other)
	{
	}

	void swap(const this_type &other)
	{
		geometry_type::swap(other);
	}

	geometry_type const &geometry() const
	{
		return *this;
	}

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

	static std::string get_type_as_string()
	{
		return "Manifold< >";
	}


	template<typename ...Args>
	auto calculate(Args &&...args) const
	DECL_RET_TYPE((calculate_policy::eval(*this, std::forward<Args>(args)...)))


	template<int I, typename ...Args>
	auto sample(Args &&...args) const
	DECL_RET_TYPE((interpolator_policy::template sample<I>(*this, std::forward<Args>(args) ...)))


	template<typename ...Args>
	auto gather(Args &&...args) const
	DECL_RET_TYPE((interpolator_policy::gather(*this, std::forward<Args>(args)...)))


	template<typename ...Args>
	void scatter(Args &&...args) const
	{
		interpolator_policy::scatter(*this, std::forward<Args>(args)...);
	}

//! @}

}; //class Manifold


}//namespace simpla

#endif /* CORE_MANIFOLD_H_ */
