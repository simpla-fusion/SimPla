/*
 * @file particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_H_
#define CORE_PARTICLE_PARTICLE_H_
#include "../application/sp_object.h"
#include "../utilities/utilities.h"
#include "../gtl/enable_create_from_this.h"
#include "../dataset/dataset.h"

namespace simpla
{

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
template<typename ... T> class Particle;

namespace _impl
{

template<typename TContainer> struct particle_container_traits
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
: public SpObject, public Engine, public enable_create_from_this<
			Particle<TM, Engine, TContainer> >
{
	typedef TM mesh_type;

	typedef Engine engine_type;

	typedef TContainer container_type;

	typedef Particle<mesh_type, engine_type, container_type> this_type;

	typedef typename container_type::value_type value_type;
	typedef typename container_type::key_type key_type;

private:

	mesh_type m_mesh_;
	std::shared_ptr<container_type> m_data_;

public:
	template<typename ...Args>
	Particle(mesh_type const & m, Args && ...args)
			: engine_type(std::forward<Args>(args)...), m_mesh_(m), m_data_(
					_impl::particle_container_traits<container_type>::create(
							m_mesh_))
	{
	}

	Particle(this_type const& other)
			: engine_type(other), m_mesh_(other.m_mesh_), m_data_(other.m_data_)
	{
	}

	Particle(this_type & other, op_split)
			: engine_type(other), m_mesh_(other.m_mesh_, op_split()), m_data_(
					other.m_data_)
	{
	}

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
	mesh_type const & mesh() const
	{
		return m_mesh_;
	}
	using engine_type::properties;
	using engine_type::print;

	size_t size() const
	{
		return m_data_->size();
	}

	template<typename ...Args>
	auto push_back(Args && ...args)
	DECL_RET_TYPE((m_data_->push_back(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto insert(Args && ...args)
	DECL_RET_TYPE((m_data_->insert(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto rehash(Args && ...args)
	DECL_RET_TYPE((m_data_->rehash(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto reserve(Args && ...args)
	DECL_RET_TYPE((m_data_->reserve(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto emplace(Args && ...args)
	DECL_RET_TYPE((m_data_->emplace(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto erase(Args && ...args)
	DECL_RET_TYPE((m_data_->erase(std::forward<Args>(args)...)))

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others)
	{
		engine_type::load(dict, std::forward<Others>(others)...);

		if (dict["DataSrc"])
		{
			UNIMPLEMENTED2("load particle from [DataSrc]");
		}
	}

	bool update()
	{
		return true;
	}

	DataSet dataset() const
	{

		DataSet res = m_data_->dump(m_mesh_.range());

		res.properties += engine_type::properties;

		return std::move(res);
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
		m_data_->foreach(m_mesh_.range(), [&](value_type & p)
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
		m_data_->foreach(m_mesh_.range(), [&](value_type & p)
		{
			for (int s = 0; s < num_of_steps; ++s)
			{
				engine_type::next_timestep(&p, t0 + dt * s, dt,
						std::forward<Args>(args)...);
			}
		});
	}

	template<typename ...Args>
	void foreach(Args && ...args)
	{
		m_data_->foreach(m_mesh_.range(), std::forward<Args>(args)...);
	}

	template<typename ...Args>
	void foreach(Args && ...args) const
	{
		m_data_->foreach(m_mesh_.range(), std::forward<Args>(args)...);
	}
}
;
}
// namespace simpla

#endif /* CORE_PARTICLE_PARTICLE_H_ */
