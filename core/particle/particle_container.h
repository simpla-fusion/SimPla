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

#include "../manifold/manifold_traits.h"
#include "../manifold/domain.h"

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

template<typename ... T> class FiberBundle;

/**
 *   `Particle<M,P>` represents a fiber bundle \f$ \pi:P\to M\f$
 *
 */



template<typename P, typename TBase>
struct Particle<P, TBase>
		: public FiberBundle<P, TBase>,
				public UnorderedSet<typename FiberBundle<P, TBase>::point_type>,
				public DistributedObject
{

private:

	typedef TBase base_manifold;

	typedef UnorderedSet<P> container_type;

	typedef FiberBundle<P, TBase> bundle_type;

	typedef Particle<P, TBase> this_type;

public:

	typedef typename FiberBundle<P, TBase>::point_type point_type;

	Particle(base_manifold const &m);

	Particle(this_type const &other);

	~Particle();

	void swap(this_type &other);

	template<typename TDict, typename ...Others> void load(TDict const &dict, Others &&...others);

	template<typename OS> OS &print(OS &os) const;

	void deploy();

	void sync();

	void wait();

	void rehash();

	DataSet dataset() const;

	void dataset(DataSet const &);

//! @}


	template<typename ...Args> void next_time_step(Args &&...args);

	template<typename ...Args> Real next_n_time_steps(size_t num_of_steps, Real t0, Real dt, Args &&...args);

private:

	struct buffer_node_s
	{
		size_t send_size;
		size_t recv_size;
		std::shared_ptr<void> send_buffer;
		std::shared_ptr<void> recv_buffer;
	};

	std::vector<buffer_node_s> m_buffer_;


};//class Particle



template<typename P, typename TBase>
Particle<P, TBase>::Particle(base_manifold const &m) :
		bundle_type(m), DistributedObject(m.comm())
{

}

template<typename P, typename TBase>
Particle<P, TBase>::Particle(this_type const &other) :
		bundle_type(other), DistributedObject(other), container_type(other)
{
}

template<typename P, typename TBase>
Particle<P, TBase>::~Particle()
{
}

template<typename P, typename TBase>
void Particle<P, TBase>::swap(this_type &other)
{
	bundle_type::swap(other);
	DistributedObject::swap(other);
	container_type::swap(other);
}

template<typename P, typename TBase>
template<typename TDict, typename ...Others>
void Particle<P, TBase>::load(TDict const &dict, Others &&...others)
{
	bundle_type::load(dict, std::forward<Others>(others)...);

	container_type::load(dict);

	if (dict["DataSet"])
	{
		DataSet ds;
		ds.load(dict["DataSet"].template as<std::string>());
		dataset(ds);
	}
}

template<typename P, typename TBase>
template<typename OS>
OS &Particle<P, TBase>::print(OS &os) const
{
	bundle_type::print(os);
	container_type::print(os);
	return os;
}

template<typename P, typename TBase>
void Particle<P, TBase>::deploy()
{

	if (bundle_type::is_valid())
	{
		return;
	}

	bundle_type::deploy();


}

template<typename P, typename TBase>
void Particle<P, TBase>::sync()
{
	auto d_type = traits::datatype<point_type>::create();

	for (auto &item :  bundle_type::mesh().template connections<VERTEX>())
	{

		buffer_node_s buffer;

		buffer.send_size = container_type::size_all(item.send_range);

		buffer.send_buffer = sp_alloc_memory(buffer.send_size * sizeof(point_type));

		point_type *data = reinterpret_cast<point_type *>(buffer.send_buffer.get());

		// FIXME need parallel optimize
		for (auto const &key : item.send_range)
		{
			for (auto const &p : container_type::operator[](key))
			{
				*data = p;
				++data;
			}
		}

		buffer.recv_size = 0;

		buffer.recv_buffer = nullptr;

		container_type::erase(item.recv_range);

		m_buffer_.push_back(buffer);

		DistributedObject::add_link_send(&item.coord_offset[0], buffer.send_size, d_type,
				&m_buffer_.back().send_buffer);
		DistributedObject::add_link_recv(&item.coord_offset[0], buffer.recv_size, d_type,
				&m_buffer_.back().recv_buffer);

	}

	DistributedObject::sync();

}

template<typename P, typename TBase>
void Particle<P, TBase>::wait()
{

};


template<typename P, typename TBase>
void Particle<P, TBase>::rehash()
{
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
	wait();
	
	container_type::rehash([&](point_type const &p)
	{
		return bundle_type::id(p);
	});

	sync();

}

template<typename P, typename TBase>
DataSet Particle<P, TBase>::dataset() const
{
	auto ds = container_type::dataset();

	DataSpace::index_type count = ds.dataspace.size();
	DataSpace::index_type offset = 0;
	DataSpace::index_type total_count = count;

	std::tie(offset, total_count) = sync_global_location(DistributedObject::comm(), count);

	DataSpace(1, &total_count).swap(ds.dataspace);

	ds.dataspace.select_hyperslab(&offset, nullptr, &count, nullptr);

	ds.dataspace.set_local_shape(&count, &offset);

	ds.properties.append(bundle_type::properties);

	return std::move(ds);
}

template<typename P, typename TBase>
void Particle<P, TBase>::dataset(DataSet const &ds)
{
	container_type::dataset(ds);

	bundle_type::properties.append(ds.properties);

	rehash();
}


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
template<typename P, typename TBase>
template<typename ...Args>
void Particle<P, TBase>::next_time_step(Args &&...args)
{

	wait();

	for (auto &item : *this)
	{
		for (auto &p : (*this)[item.first])
		{
			bundle_type::move(&p, std::forward<Args>(args)...);
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

template<typename P, typename TBase>
template<typename ...Args>
Real Particle<P, TBase>::next_n_time_steps(size_t num_of_steps, Real t0, Real dt, Args &&...args)
{

	wait();

	for (auto &item : *this)
	{
		for (auto &p : (*this)[item.first])
		{
			for (int s = 0; s < num_of_steps; ++s)
			{
				bundle_type::move(&p, t0 + dt * s, dt, std::forward<Args>(args)...);
			}
		}
	}
}
}  // namespace simpla

#endif /* CORE_PARTICLE_PARTICLE_CONTAINER_H_ */
