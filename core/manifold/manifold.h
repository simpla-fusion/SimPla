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
#include "../utilities/utilities.h"
namespace simpla
{

/**
 * \addtogroup  manifold Manifold
 *    \brief   Discrete spatial-temporal space \see @ref manifold_concept
 *    @file manifold.md
 */

template<typename > class FiniteDiffMethod;
template<typename > class InterpolatorLinear;
template<typename, size_t> class Domain;
template<typename ...> class _Field;
template<typename, typename, typename > class Expression;

/**
 *  \ingroup manifold
 *  \brief Manifold
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

	typedef std::shared_ptr<this_type> holder_type;

	typedef TG geometry_type;
	typedef Policy1<geometry_type> policy1;
	typedef Policy2<geometry_type> policy2;
	typedef typename geometry_type::topology_type topology_type;

	typedef typename geometry_type::coordinates_type coordinates_type;
	typedef typename geometry_type::id_type id_type;
//	typedef typename geometry_type::iterator iterator;
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
	static holder_type create(Args &&... args)
	{
		return std::make_shared<this_type>(std::forward<Args>(args)...);
	}

	this_type & operator=(this_type const &) = delete;

	template<typename TDict>
	bool load(TDict const & dict)
	{
		VERBOSE << "Load Manifold" << std::endl;

		if (!(topology_type::load(dict["Topology"])
				&& geometry_type::load(dict["Geometry"])))
		{
			RUNTIME_ERROR("ERROR: Load Manifold failed!");

			return false;
		}
		return true;
	}
	bool update()
	{
		return topology_type::update() && geometry_type::update();
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
