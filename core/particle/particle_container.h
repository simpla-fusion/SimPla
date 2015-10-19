/**
 * @file particle_container.h
 *
 *  Created on: 2015-3-26
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_CONTAINER_H_
#define CORE_PARTICLE_PARTICLE_CONTAINER_H_

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../dataset/dataset.h"
#include "../dataset/datatype.h"
#include "../gtl/utilities/log.h"
#include "../gtl/utilities/memory_pool.h"
#include "../gtl/enable_create_from_this.h"
#include "../gtl/primitives.h"
#include "../gtl/properties.h"
#include "../gtl/iterator/sp_iterator.h"

#include "../gtl/containers/unordered_set.h"
#include "../parallel/distributed.h"
#include "../parallel/distributed_unordered_set.h"
#include "../manifold/manifold_traits.h"
#include "../manifold/domain.h"
#include "../manifold/fiber_bundle.h"
/** @ingroup physical_object
*  @addtogroup particle Particle
*  @{
*	  @brief  @ref particle  is an abstraction from  physical particle or "phase-space sample".
*	  @details
* ## Summary
*  - @ref particle is used to  describe trajectory in  @ref phase_space_7d  .
*  - @ref particle is used to  describe the behavior of  discrete samples of
*    @ref phase_space_7d function  \f$ f\left(t,x,y,z,v_x,v_y,v_z \right) \f$.
*  - @ref particle is a @ref container;
*  - @ref particle is @ref splittable;
*  - @ref particle is a @ref field
* ### Data Structure
*  -  @ref particle is  `unorder_set<Point_s>`
*
* ## Requirements
*- The following table lists the requirements of a Particle type  '''P'''
*	Pseudo-Signature    | Semantics
* -------------------- |----------
* ` struct Point_s `   | data  type of sample point
* ` P( ) `             | Constructor
* ` ~P( ) `            | Destructor
* ` void  next_time_step(dt, args ...) const; `  | push  particles a time interval 'dt'
* ` void  next_time_step(num_of_steps,t0, dt, args ...) const; `  | push  particles from time 't0' to 't1' with time step 'dt'.
* ` flush_buffer( ) `  | flush input buffer to internal data container
*
*- @ref particle meets the requirement of @ref container,
* Pseudo-Signature                 | Semantics
* -------------------------------- |----------
* ` push_back(args ...) `          | Constructor
* ` foreach(TFun const & fun)  `   | Destructor
* ` dataset dump() `               | dump/copy 'data' into a DataSet
*
*- @ref particle meets the requirement of @ref physical_object
*   Pseudo-Signature           | Semantics
* ---------------------------- |----------
* ` print(std::ostream & os) ` | print decription of object
* ` update() `                 | update internal data storage and prepare for execute 'next_time_step'
* ` sync()  `                  | sync. internal data with other processes and threads
*
*
* ## Description
* @ref particle   consists of  @ref particle_container and @ref particle_engine .
*   @ref particle_engine  describes the individual behavior of one sample. @ref particle_container
*	  is used to manage these samples.
*
*
* ## Example
*
*  @}
*/
namespace simpla
{


template<typename ... T> class Particle;


template<typename ... T> class Distributed;


namespace particle
{

template<typename P, typename M>
class FiberBundle
{


};
}

/**
 *   `Particle<M,P>` represents a fiber bundle \f$ \pi:P\to M\f$
 *
 */
template<typename P, typename M>
class Particle<P, M>
		: public FiberBundle<P, M>,
				public Distributed<UnorderedSet<typename P::point_type, typename particle::FiberBundle<P, M>::hasher> >
{
public:

	typedef M base_manifold_type;

	static constexpr size_t iform = VOLUME;

	typedef Domain<base_manifold_type, std::integral_constant<int, iform>> domain_type;

	typedef Distributed<UnorderedSet<typename P::point_type, typename particle::FiberBundle<P, M>::hasher> > container_type;

	typedef Particle<P, M> this_type;

	typedef typename P::point_type point_type;

//	typedef typename mesh_type::index_type index_type;
//	typedef typename mesh_type::id_type id_type;
//	typedef typename mesh_type::point_type point_type;
//	typedef typename mesh_type::vector_type vector_type;


private:

	domain_type m_domain_;

	base_manifold_type const &m_mesh_;

public:

	Particle(domain_type const &d)
			: m_domain_(d), m_mesh_(d.mesh())
	{
	}

	Particle(this_type const &other) :
			container_type(other),
			m_domain_(other.m_domain_), m_mesh_(other.m_mesh_)
	{
	}


	~Particle()
	{
	}

	Properties properties;


	this_type const &self() const
	{
		return *this;
	}


	mesh_type const &mesh() const
	{
		return m_mesh_;
	}

	domain_type const &domain() const
	{
		return m_domain_;
	}

	template<typename TDict, typename ...Others>
	void load(TDict const &dict, Others &&...others)
	{
		engine_type::load(dict, std::forward<Others>(others)...);

		if (dict["DataSrc"])
		{
			UNIMPLEMENTED2("load particle from [DataSrc]");
		}
	}

	template<typename OS>
	OS &print(OS &os) const
	{
		engine_type::print(os);
		os << " num= " << container_type::size() << ",";
		return os;
	}


	bool is_valid() const
	{
		return engine_type::is_valid();
	}

	void deploy();


	bool is_ready() const
	{
		return true;
	}




//! @}

/**
 *
 * @param args arguments
 *
 * - Semantics
 @code
 for( Point_s & point: all particle)
 {
 engine_type::next_time_step(& point,std::forward<Args>(args)... );
 }
 @endcode
 *
 */
	template<typename ...Args>
	void next_time_step(Args &&...args)
	{

		wait();

		for (auto &item : *this)
		{
			for (auto &p : (*this)[item.first])
			{
				engine_type::next_time_step(&p, std::forward<Args>(args)...);
			}
		}
	}

/**
 *
 * @param num_of_steps number of time steps
 * @param t0 start time point
 * @param dt delta time step
 * @param args other arguments
 * @return t0+num_of_steps*dt
 *
 *-Semantics
 @code
 for(s=0;s<num_of_steps;++s)
 {
 for( Point_s & point: all particle)
 {
 engine_type::next_time_step(& point,t0+s*dt,dt,std::forward<Args>(args)... );
 }
 }
 return t0+num_of_steps*dt;
 @endcode
 */
	template<typename ...Args>
	Real next_n_time_steps(size_t num_of_steps, Real t0, Real dt,
			Args &&...args)
	{

		wait();

		for (auto &item : *this)
		{
			for (auto &p : (*this)[item.first])
			{
				for (int s = 0; s < num_of_steps; ++s)
				{
					engine_type::next_time_step(&p, t0 + dt * s, dt,
							std::forward<Args>(args)...);
				}
			}
		}
	}

//	void rehash()
//	{
//		container_type::rehash();
//		point_type xmin, xmax;
//		std::tie(xmin, xmax) = m_domain_.mesh().extents();
//		point_type d;
//		d = xmax - xmin;
//
//		for (auto &item : *this)
//		{
//			point_type x0;
//			x0 = m_domain_.mesh().point(item.first);
//
////			if (!m_mesh_.in_box(x0, xmin, xmax))
////			{
////				for (auto &p : (*this)[item.first])
////				{
////					point_type x;
////					Vec3 v;
////					Real f;
////					std::tie(x, v, f) = engine_type::pull_back(p);
////
////					x[0] += std::fmod(x[0] - xmin[0] + d[0], d[0]);
////					x[1] += std::fmod(x[1] - xmin[1] + d[1], d[1]);
////					x[2] += std::fmod(x[2] - xmin[2] + d[2], d[2]);
////
////					engine_type::push_forward(x, v, f, &p);
////				}
////			}
//
//		}
//		container_type::rehash();
//
//
//	}


};//class Particle

template<typename TDomain, typename Engine>
void Particle<TDomain, Engine>::deploy()
{

	engine_type::deploy();

	properties.append(engine_type::properties);

	m_mesh_.for_each_ghost_range([&](int const coord_offset[3], range_type const
	&send_range, range_type const &recv_range)
	{
		container_type::add_link(coord_offset, send_range, recv_range);
	}
	);

}

template<typename TDomain, typename Engine, typename TContainer>
DataSet Particle<TDomain, Engine, TContainer>::dataset() const
{

	DataSet res;

	res.datatype = traits::datatype<value_type>::create();
	res.properties = properties;

	index_type count = 0;

	m_domain_.for_each([&](id_type const &s)
	{
		auto it = container_type::find(s);
		if (it != container_type::end())
		{
			count += std::distance(it->second.begin(), it->second.end());
		}
	});

	res.data = sp_alloc_memory(count * sizeof(value_type));

	value_type *p = reinterpret_cast<value_type *>(res.data.get());

	//TODO need parallel optimize

	m_domain_.for_each([&](id_type const &s)
	{
		auto it = container_type::find(s);

		if (it != container_type::end())
		{
			for (auto const &pit : it->second)
			{
				*p = pit;
				++p;
			}
		}
	});

//		ASSERT(std::distance(data.get(), p) == count);

	index_type offset = 0;
	index_type total_count = count;

	std::tie(offset, total_count) = sync_global_location(count);

	DataSpace(1, &total_count).swap(res.dataspace);

	res.dataspace.select_hyperslab(&offset, nullptr, &count, nullptr);

	res.dataspace.set_local_shape(&count, &offset);

	VERBOSE << "dump particles [" << count << "/" << total_count << "] "
			<< std::endl;

	return std::move(res);
}

}  // namespace simpla

#endif /* CORE_PARTICLE_PARTICLE_CONTAINER_H_ */
