/*
 * mainfold.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef MAINFOLD_H_
#define MAINFOLD_H_

#include <utility>
#include "../utilities/sp_type_traits.h"
namespace simpla
{

template<typename > class FiniteDiffMethod;
template<typename > class InterpolatorLinear;

template<typename TG, //
		template<typename > class DiffSchemePolicy = FiniteDiffMethod, //
		template<typename > class InterpoloatorPolicy = InterpolatorLinear>
class Manifold: public TG
{
	typedef Manifold<TG, DiffSchemePolicy, InterpoloatorPolicy> this_type;
	typedef TG geometry_type;
	typedef typename geometry_type::topology_type topology_type;
	typedef DiffSchemePolicy<this_type> diff_policy;
	typedef InterpoloatorPolicy<this_type> interpoloator_policy;

	template<unsigned int IFORM = VERTEX>
	Domain<this_type, IFORM> domain() const
	{
		return std::move(Domain<this_type, IFORM>(*this));
	}
	template<unsigned int IFORM, typename TV> using
	field=Field<Domain<this_type,IFORM>,storage_policy<TV> >;

	template<typename TOP, typename TL, typename TR, typename TI>
	auto eval(TL const & l, TR const &r, TI const &s) const
	DECL_RET_TYPE((diff_policy::eval(TOP(),*this, l, r, s)))

	template<typename TOP, typename TL, typename TI>
	auto eval(TL const & l, TI const &s) const
	DECL_RET_TYPE((diff_policy::eval(TOP(),*this, l, s)))

	template<typename TL, typename ...Others>
	auto scatter(TL & l, Others &&... others) const
	DECL_RET_TYPE((interpoloator_policy::scatter(*this,
							l, std::forward<Others>(others)...)))

	template<typename TL, typename ...Others>
	void scatter(TL & l, Others &&... others) const
	{
		return ((interpoloator_policy::scatter(*this, l,
				std::forward<Others>(others)...)));
	}

};

}
// namespace simpla

#endif /* MAINFOLD_H_ */
