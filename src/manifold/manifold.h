/*
 * manifold.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef MANIFOLD_H_
#define MANIFOLD_H_

#include <utility>
#include <vector>
#include "../utilities/sp_type_traits.h"
namespace simpla
{

template<typename > class FiniteDiffMethod;
template<typename > class InterpolatorLinear;
template<typename, unsigned int> class Domain;
template<typename, typename > class Field;

enum GeometryFormTypeID
{
	VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3
};

template<typename TG, //
		template<typename > class DiffSchemePolicy = FiniteDiffMethod, //
		template<typename > class InterpoloatorPolicy = InterpolatorLinear>
class Manifold: public TG
{
public:
	Manifold() = default;
	~Manifold() = default;

	typedef Manifold<TG, DiffSchemePolicy, InterpoloatorPolicy> this_type;
	typedef TG geometry_type;

	typedef typename geometry_type::topology_type topology_type;
	typedef DiffSchemePolicy<this_type> diff_policy;
	typedef InterpoloatorPolicy<this_type> interpoloator_policy;

	typedef typename geometry_type::coordinates_type coordiantes_type;
	typedef typename geometry_type::index_type index_type;

	template<unsigned int IFORM = VERTEX>
	Domain<this_type, IFORM> domain() const
	{
		return std::move(Domain<this_type, IFORM>(*this));
	}
	template<unsigned int IFORM, typename TV> using
	field=Field<Domain<this_type,IFORM>,std::vector<TV> >;

	template<typename TF> TF make_field() const
	{
		return std::move(TF(domain<TF::iform>()));
	}

	template<typename TOP, typename ...Args>
	auto calculus(Args && ...args) const
	DECL_RET_TYPE((
					diff_policy::calculus(TOP(),*this,
							std::forward<Args>(args)...)))

	template<typename ...Args>
	auto gather(Args &&... args) const
	DECL_RET_TYPE((
					interpoloator_policy::gather_(*this,
							std::forward<Args>(args)...)
			))

	template<typename ...Args>
	auto scatter(Args &&... args) const
	DECL_RET_TYPE((
					interpoloator_policy::gather_(*this,
							std::forward<Args>(args)...)
			))

};

}
// namespace simpla

#endif /* MANIFOLD_H_ */
