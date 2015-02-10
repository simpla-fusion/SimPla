/*
 * particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_
#include <unordered_set>
namespace simpla
{

template<typename TGeo, typename TPoint_s>
struct particle_hasher
{
	typedef TGeo geometry_type;
	typedef typename geometry_type::id_type key_type;
	typedef TPoint_s value_type;

	geometry_type const & m_geo_;

	particle_hasher(geometry_type const & geo) :
			m_geo_(geo)
	{
	}
	~particle_hasher()
	{
	}

	constexpr key_type operator()(value_type const & p) const
	{
		return m_geo_.coordinates_to_id(p.x);
	}
}
;

template<typename T> void sync(T *);

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

template<typename TG, typename Engine, typename ...Others>
class Particle: public Engine,
		public sp_sorted_set<typename Engine::Point_s,
				particle_hasher<TG, typename Engine::Point_s> >,
		public enable_create_from_this<Particle<Engine, Others...>>
{
	typedef TG geometry_type;

	typedef Engine engine_type;

	typedef Particle<geometry_type, engine_type, Others...> this_type;

	typedef typename engine_type::Point_s value_type;

	typedef typename geometry_type::id_type key_type;

private:

	geometry_type const & m_geo_;

	typedef sp_sorted_set<value_type, hasher> container_type;

	typedef typename container_type::bucket_type bucket_type;

public:
	Particle(geometry_type const & geo) :
			container_type(hasher(geo)), m_geo_(geo)
	{
	}

	Particle(this_type const& other) = delete;

	~Particle()
	{
	}
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
	geometry_type const & geometry() const
	{
		return m_geo_;
	}
	using engine_type::properties;
	using engine_type::print;

	using container_type::push_back;
	using container_type::insert;
	using container_type::emplace;
	using container_type::rehash;
	using container_type::reserver;
	using container_type::erase;

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others)
	{
		engine_type::load(dict);
	}

	bool update()
	{
		return true;
	}

	void sync()
	{
		simpla::sync(this);
	}

	DataSet dataset() const
	{
		size_t num;

		std::shared_ptr<value_type> data;

		std::tie(num, data) = m_data_.dump(m_geo_.select<VERTEX>());

		return DataSet(
		{ data, DataType::create<value_type>(), DataSpace(1, &num),
				engine_type::properties });
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
		for (auto & item : m_data_)
		{
			for (auto & p : item.second)
			{
				engine_type::next_timestep(&p, std::forward<Args>(args)...);

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
		for (auto & item : m_data_)
		{
			for (auto & p : item.second)
			{
				for (int s = 0; s < num_of_steps; ++s)
				{
					engine_type::next_timestep(&p, t0 + dt * s, dt,
							std::forward<Args>(args)...);
				}
			}
		}
	}

	template<typename TRange, typename TFun, typename ...Args>
	void foreach(TRange const & range, TFun const & fun, Args && ...)
	{
		for (auto const & s : range)
		{
			auto it = m_data_.find(s);
			if (it != m_data_.end())
			{
				for (auto & p : it->second)
				{
					fun(p, std::forward<Args>(args)...);
				}
			}
		}
	}

	template<typename TRange, typename TFun, typename ...Args>
	void foreach(TRange const & range, TFun const & fun, Args && ...) const
	{
		for (auto const & s : range)
		{
			auto it = m_data_.find(s);
			if (it != m_data_.end())
			{
				for (auto const & p : it->second)
				{
					fun(p, std::forward<Args>(args)...);
				}
			}
		}
	}
};
}
// namespace simpla

#endif /* PARTICLE_H_ */
