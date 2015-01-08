/*
 * particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

/**
 *  @addtogroup particle Particle
 *  @{
 *	  @brief  @ref particle  is an abstraction from  physical particle or "phase-space sample".
 *	  @details
 * ## Summary
 *  - @ref particle is used to  describe trajectory in  @ref phase_space_7d  .
 *  - @ref particle is used to  describe the behavior of  discrete samples of
 *   @ref phase_space_7d function  \f$ f\left(t,x,y,z,v_x,v_y,v_z \right) \f$.
 *
 *
 *
 * ## Requirements
 *- The following table lists the requirements of a Particle type  '''P'''
 *	Pseudo-Signature   |Semantics
 * ------------- |----------
 * \code struct Point_s \endcode | data  type of sample point
 * \code P( ) \endcode    | Constructor
 * \code ~P( ) \endcode  | Destructor
 * \code void  next_timestep(dt, args ...) const; \endcode  | push  particles a time interval 'dt'
 * \code void  next_timestep(t0,t1,dt, args ...) const; \endcode  | push  particles from time 't0' to 't1' with time step 'dt'.
 * \code flush_buffer( ) \endcode  | flush input buffer to internal data container
 *
 *- @ref particle meets the requirement of @ref container,
 * Pseudo-Signature   |Semantics
 * ------------- |----------
 * \code push_back(args ...) \endcode    | Constructor
 * \code foreach(TFun const & fun)  \endcode  | Destructor
 * \code dataset() \endcode |  data interface of container
 *
 *- @ref particle meets the requirement of @ref physical_object
 * Pseudo-Signature   |Semantics
 * ------------- |----------
 * \code print(std::ostream & os) \endcode | print decription of object
 * \code update() \endcode | update internal data storage and prepare for execute 'next_timestep'
 * \code sync()  \endcode  | sync. internal data with other processes and threads
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
#endif /* PARTICLE_H_ */
