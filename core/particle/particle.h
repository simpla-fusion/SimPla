/*
 * particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

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
 *
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
 * ` dataset() `                    | data interface of container
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
//
//template<typename Engine, typename ...Others>
//class Particle: public PhysicalObject,
//				public Engine,
//				public enable_create_from_this<Particle<Engine, Others...>>
//
//{
//	typedef Engine engine_type;
//
//	typedef Particle<engine_type, Others...> this_type;
//
//	typedef typename engine_type::Point_s Point_s;
//
//	Particle();
//
//	Particle(this_type const&);
//
//	~Particle();
//
//	using engine_type::properties;
//
//	std::ostream& print(std::ostream & os) const
//	{
//		engine_type::print(os);
//		return os;
//	}
//
//	template<typename TDict, typename ...Others>
//	void load(TDict const & dict, Others && ...others);
//
//	bool update();
//
//	void sync();
//
//	DataSet dataset() const;
//
//	static std::string get_type_as_string_staic()
//	{
//		return engine_type::get_type_as_string();
//	}
//	std::string get_type_as_string() const
//	{
//		return get_type_as_string_staic();
//	}
//
//	//! @name   @ref splittable
//	//! @{
//
//	Particle(this_type&, split);
//
//	bool empty() const;
//
//	bool is_divisible() const;
//
//	//! @}
//	/**
//	 *
//	 * @param args arguments
//	 *
//	 * - Semantics
//	 @code
//	 for( Point_s & point: all particle)
//	 {
//	 engine_type::next_timestep(& point,std::forward<Args>(args)... );
//	 }
//	 @endcode
//	 *
//	 */
//	template<typename ...Args>
//	void next_timestep(Args && ...args);
//
//	/**
//	 *
//	 * @param num_of_steps number of time steps
//	 * @param t0 start time point
//	 * @param dt delta time step
//	 * @param args other arguments
//	 * @return t0+num_of_steps*dt
//	 *
//	 *-Semantics
//	 @code
//	 for(s=0;s<num_of_steps;++s)
//	 {
//	 for( Point_s & point: all particle)
//	 {
//	 engine_type::next_timestep(& point,t0+s*dt,dt,std::forward<Args>(args)... );
//	 }
//	 }
//	 return t0+num_of_steps*dt;
//	 @endcode
//	 */
//	template<typename ...Args>
//	Real next_n_timesteps(size_t num_of_steps, Real t0, Real dt,
//			Args && ...args);
//
//	/**
//	 *  push_back and emplace will invalid data in the cache
//	 * @param args
//	 */
//	template<typename ...Args>
//	void push_back(Args && ...args);
//
//	template<typename ...Args>
//	void emplac_back(Args && ...args);
//
//	template<typename TFun, typename ...Args>
//	void foreach(TFun const & fun, Args && ...);
//
//};

}
// namespace simpla

#endif /* PARTICLE_H_ */
