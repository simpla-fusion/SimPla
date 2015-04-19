/**
 * @file  manifold.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef CORE_MESH_STRUCTURED_MANIFOLD_H_
#define CORE_MESH_STRUCTURED_MANIFOLD_H_

#include <stddef.h>

#include "../gtl/type_traits.h"

namespace simpla
{

template<typename > class FiniteDiffMethod;
template<typename > class InterpolatorLinear;

/**
 *  \ingroup manifold
 *  \brief manifold
 */
template<typename TG, // Geometric space, mesh
		typename CalculusPolicy = FiniteDiffMethod<TG>, // difference scheme
		typename InterpolatorPlolicy = InterpolatorLinear<TG> // interpolation formula
>
class Manifold: public TG
{
public:
	typedef Manifold<TG, CalculusPolicy, InterpolatorPlolicy> this_type;

	typedef TG geometry_type;
	typedef typename geometry_type::id_type id_type;
	typedef typename geometry_type::topology_type topology_type;
	typedef typename geometry_type::index_tuple index_tuple;
	typedef typename geometry_type::coordinates_type coordinates_type;
	static constexpr size_t ndims = geometry_type::ndims;

	typedef CalculusPolicy calculate_policy;
	typedef InterpolatorPlolicy interpolatpr_policy;

public:

	Manifold()
	{
	}

	Manifold(this_type const & other) :
			geometry_type(other)
	{
	}

	~Manifold() = default;

	void swap(this_type & other)
	{
		geometry_type::swap(other);
	}
	this_type & operator=(const this_type& other)
	{
		this_type(other).swap(*this);
		return *this;
	}

	template<size_t IFORM>
	auto domain() const
	DECL_RET_TYPE((Domain<this_type, IFORM>(*this)))

public:

	template<typename ...Args>
	auto calculate(Args && ...args) const
	DECL_RET_TYPE(( calculate_policy::calculate(
							*this,std::forward<Args>(args)...)))
	template<size_t iform, typename TF>
	auto sample(TF const & v, id_type s) const
	DECL_RET_TYPE(interpolatpr_policy::template sample<iform>(*this ,s,v))

	template<typename ...Args>
	auto gather(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::gather(
							*this,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::scatter(
							*this,std::forward<Args>(args)...)))

};

}
// namespace simpla

#endif /* CORE_MESH_STRUCTURED_MANIFOLD_H_ */
