/*
 * manifold.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef MANIFOLD_H_
#define MANIFOLD_H_
#include <memory>
#include <utility>
#include <vector>
#include <ostream>
#include "../utilities/sp_type_traits.h"
#include "../utilities/primitives.h"
#include "../utilities/log.h"
namespace simpla
{

template<typename > class FiniteDiffMethod;
template<typename > class InterpolatorLinear;
template<typename, size_t> class Domain;
template<typename ...> class _Field;
template<typename, typename, typename > class Expression;

/**
 *
 *  \brief Manifold
 *
 *
 */

template<typename TG, //
		template<typename > class Policy1 = FiniteDiffMethod, //
		template<typename > class Policy2 = InterpolatorLinear>
class Manifold: public TG,
		public Policy1<TG>,
		public Policy2<TG>,
		public std::enable_shared_from_this<Manifold<TG, Policy1, Policy2>>
{
public:

	typedef Manifold<TG, Policy1, Policy2> this_type;
	typedef TG geometry_type;
	typedef Policy1<geometry_type> policy1;
	typedef Policy2<geometry_type> policy2;
	typedef typename geometry_type::topology_type topology_type;
	typedef typename geometry_type::coordinates_type coordinates_type;
	typedef typename geometry_type::index_type index_type;
	typedef typename geometry_type::compact_index_type compact_index_type;
	typedef typename geometry_type::iterator iterator;
	typedef typename geometry_type::scalar_type scalar_type;

	static constexpr size_t ndims = topology_type::ndims;

	template<typename ...Args>
	Manifold(Args && ... args) :
			geometry_type(std::forward<Args>(args)...)
	{
		policy1::geometry(this);
		policy2::geometry(this);
	}

	~Manifold() = default;

	Manifold(this_type const & r) = delete;

	template<typename ...Args>
	static std::shared_ptr<this_type> create(Args &&... args)
	{
		return std::make_shared<this_type>(std::forward<Args>(args)...);
	}

	this_type & operator=(this_type const &) = delete;

	template<typename TDict>
	void load(TDict const &)
	{

	}
	bool update()
	{
		return true;
	}

};

template<typename TM, typename ...Args>
std::shared_ptr<TM> make_manifold(Args && ...args)
{
	return std::make_shared<TM>(std::forward<Args>(args)...);
}

}
// namespace simpla

#endif /* MANIFOLD_H_ */
