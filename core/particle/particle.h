/*
 * @file particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_H_
#define CORE_PARTICLE_PARTICLE_H_

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../application/sp_object.h"
#include "../dataset/dataset.h"
#include "../utilities/log.h"
#include "../utilities/memory_pool.h"
#include "../gtl/enable_create_from_this.h"
#include "../gtl/primitives.h"
#include "../gtl/properties.h"
#include "../gtl/iterator/sp_iterator.h"
#include "../parallel/mpi_update.h"
#include "../parallel/mpi_aux_functions.h"

namespace simpla
{

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
 * ` void  next_timestep(dt, args ...) const; `  | push  particles a time interval 'dt'
 * ` void  next_timestep(num_of_steps,t0, dt, args ...) const; `  | push  particles from time 't0' to 't1' with time step 'dt'.
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
 * ` update() `                 | update internal data storage and prepare for execute 'next_timestep'
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
template<typename ... T> class Particle;

namespace _impl
{

template<typename TContainer>
struct particle_container_traits
{
	template<typename ...Args>
	static std::shared_ptr<TContainer> create(Args && ...args)
	{
		return std::make_shared<TContainer>();
	}
};

}  // namespace _impl

template<typename TM, typename Engine, typename TContainer>
class Particle<TM, Engine, TContainer> //
:	public SpObject,
	public Engine,
	public TContainer,
	public enable_create_from_this<Particle<TM, Engine, TContainer> >
{
public:
	typedef TM mesh_type;

	typedef Engine engine_type;

	typedef TContainer container_type;

	typedef Particle<mesh_type, engine_type, container_type> this_type;

	typedef typename container_type::value_type value_type;

private:
	mesh_type m_mesh_;
public:

	Particle(mesh_type const & m)
			: m_mesh_(m)
	{
	}

	Particle(this_type const& other)
			: engine_type(other), container_type(other), m_mesh_(other.m_mesh_)
	{
	}

	template<typename ... Args>
	Particle(this_type & other, Args && ...args)
			: engine_type(other), container_type(other,
					std::forward<Args>(args)...), m_mesh_(other.m_mesh_)
	{
	}

	~Particle()
	{
	}

	using SpObject::properties;

	this_type & self()
	{
		return *this;
	}
	this_type const& self() const
	{
		return *this;
	}

	static std::string get_type_as_string_static()
	{
		return engine_type::get_type_as_string();
	}
	std::string get_type_as_string() const
	{
		return get_type_as_string_static();
	}
	mesh_type const & mesh() const
	{
		return m_mesh_;
	}
	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others)
	{
		engine_type::load(dict, std::forward<Others>(others)...);

		if (dict["DataSrc"])
		{
			UNIMPLEMENTED2("load particle from [DataSrc]");
		}
	}

	void deploy()
	{
		engine_type::deploy();

		SpObject::properties.append(engine_type::properties);
	}

	void sync()
	{
		std::vector<mpi_send_recv_buffer_s> send_recv_buffer;

		auto const & ghost_list = m_mesh_.ghost_shape();

		for (auto const & item : ghost_list)
		{
			mpi_send_recv_buffer_s send_recv_s;

			std::tie(send_recv_s.dest, send_recv_s.send_tag) = get_mpi_tag(
					&item.coord_shift[0]);

			//   collect send data

			auto send_range = m_mesh_.select(item.send_offset, item.send_count);

			size_t send_num = container_type::size(send_range);

			send_recv_s.send_size = send_num * sizeof(value_type);

			send_recv_s.send_data = sp_alloc_memory(send_recv_s.send_size);

			value_type *data =
					reinterpret_cast<value_type*>(send_recv_s.send_data.get());

			// FIXME need parallel optimize
			for (auto const & key : send_range)
			{
				for (auto const & p : container_type::operator[](
						m_mesh_.hash(key)))
				{
					*data = p;
					++data;
				}
			}

			//  clear ghosts cell
			auto recv_range = m_mesh_.select(item.recv_offset, item.recv_count);

			container_type::erase(recv_range);

			send_recv_s.recv_size = 0;
			send_recv_s.recv_data = nullptr;
			send_recv_buffer.push_back(std::move(send_recv_s));

		}

		sync_update_varlength(&send_recv_buffer);

		for (auto const & item : send_recv_buffer)
		{
			size_t count = item.recv_size / sizeof(value_type);

			value_type *data =
					reinterpret_cast<value_type*>(item.recv_data.get());

			container_type::insert(data, data + count);
		}

	}

	template<typename TRange>
	DataSet dataset(TRange const & p_range) const
	{
		DataSet res;

		res.datatype = DataType::create<value_type>();
		res.properties = SpObject::properties;

		size_t count = 0;

		for (auto const & key : p_range)
		{
			auto it = container_type::find(key);
			if (it != container_type::end())
			{
				count += std::distance(it->second.begin(), it->second.end());
			}
		}

		res.data = sp_alloc_memory(count * sizeof(value_type));

		value_type * p = reinterpret_cast<value_type *>(res.data.get());

		//TODO need parallel optimize

		auto back_insert_it = back_inserter(p);

		for (auto const & key : p_range)
		{
			auto it = container_type::find(key);

			if (it != container_type::end())
			{
				std::copy(it->second.begin(), it->second.end(), back_insert_it);
			}
		}

		ASSERT(std::distance(data.get(), back_insert_it.get()) == count);

		size_t offset = 0;
		size_t total_count = count;

		std::tie(offset, total_count) = sync_global_location(count);

		res.dataspace = //
				DataSpace(1, &total_count) //
				.select_hyperslab(&offset, nullptr, &count, nullptr) //
				.create_distributed_space();

//		int ndim;
//		nTuple<size_t, 3> l_offset, l_count, l_dims;
//		l_offset = 0;
//		l_count = 0;
//		l_dims = 0;
//
//		std::tie(ndim, l_dims, l_offset, std::ignore, l_count, std::ignore) =
//				dataspace.shape();
//
//		CHECK(ndim);
//		CHECK(l_dims);
//		CHECK(l_offset);
//		CHECK(l_count);
//
//		nTuple<size_t, 3> g_offset, g_count, g_dims;
//		g_offset = 0;
//		g_count = 0;
//		g_dims = 0;
//
//		std::tie(ndim, g_dims, g_offset, std::ignore, g_count, std::ignore) =
//				dataspace.global_shape();
//
//		CHECK(ndim);
//		CHECK(g_dims);
//		CHECK(g_offset);
//		CHECK(g_count);

		return std::move(res);
	}

	DataSet dataset() const
	{
		return std::move(dataset(m_mesh_.range()));
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
	 engine_type::next_timestep(& point,std::forward<Args>(args)... );
	 }
	 @endcode
	 *
	 */
	template<typename ...Args>
	void next_timestep(Args && ...args)
	{
		container_type::foreach(m_mesh_.range(), [&](value_type & p)
		{
			engine_type::next_timestep(&p, std::forward<Args>(args)...);
		});
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
	 engine_type::next_timestep(& point,t0+s*dt,dt,std::forward<Args>(args)... );
	 }
	 }
	 return t0+num_of_steps*dt;
	 @endcode
	 */
	template<typename ...Args>
	Real next_n_timesteps(size_t num_of_steps, Real t0, Real dt,
			Args && ...args)
	{
		container_type::foreach(m_mesh_.range(), [&](value_type & p)
		{
			for (int s = 0; s < num_of_steps; ++s)
			{
				engine_type::next_timestep(&p, t0 + dt * s, dt,
						std::forward<Args>(args)...);
			}
		});
	}

}
;
}
// namespace simpla

#endif /* CORE_PARTICLE_PARTICLE_H_ */
