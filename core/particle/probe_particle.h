/*
 * probe_particle.h
 *
 *  Created on: 2014年11月18日
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PROBE_PARTICLE_H_
#define CORE_PARTICLE_PROBE_PARTICLE_H_

#include <iostream>
#include <memory>
#include <string>

#include "../containers/sp_iterator_sequence.h"
#include "../data_interface/data_set.h"
#include "../physics/physical_object.h"
#include "../utilities/log.h"
#include "../utilities/primitives.h"
#include "../parallel/parallel.h"

namespace simpla
{

/**
 *
 * @ingroup particle_container
 *
 * @brief  ProbeParticle is a container of particle trajectory
 *
 *  It can cache the history of particle position.
 *
 *  * function next_timestep(Point * p, Real dt, Args && ...)
 *
 *    p - m0  particle position  at m0 steps before
 *
 *    p - 1   particle position  at last time step
 *
 *    p       particle position  at current time step
 *
 *    p + 1   particle position  at next time step
 *
 *    p + 2   predicate particle position  after next two time steps
 *
 *    p + m1  predicate particle position  after next m1 time steps
 *
 *    default: m0=m1=0
 *
 *
 *  Requirement:
 *    engine_type::next_timestep(Point_s * p, others...);
 *
 *  - if  engine_type::memory_length  is not defined
 *     p point the "Particle" at current time step
 *
 *  - if engine_type::memory_length = m
 *    p-1,p-2,... , p-m are valid and point to "Particles" at previous m steps
 *
 */
template<typename Engine>
struct ProbeParticle: public PhysicalObject, public Engine
{

	//! @name   Basic usage
	//! @{
	typedef Engine engine_type;

	typedef ProbeParticle<engine_type> this_type;

	typedef typename engine_type::Point_s Point_s;

	//! Constructor
	template<typename ...Others>
	ProbeParticle(Others && ...);

	//! Destroy
	~ProbeParticle();

	using engine_type::properties;

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others);

	static std::string get_type_as_string_staic()
	{
		return engine_type::get_type_as_string();
	}
	std::string get_type_as_string() const
	{
		return get_type_as_string_staic();
	}

	template<typename ...Args>
	void next_timestep(Args && ...args);

	template<typename ...Args>
	void next_n_timesteps(size_t num_of_steps, Real t0, Real dt,
			Args && ...args);

	std::ostream& print(std::ostream & os) const
	{
		engine_type::print(os);
		return os;
	}

	template<typename ...Args>
	void push_back(Args && ...args)
	{
		buffer.push_back(std::forward<Args>(args)...);
	}

	template<typename ...Args>
	void emplac_back(Args && ...args)
	{
		buffer.emplac_back(std::forward<Args>(args)...);
	}

	bool update();

	void sync();

	//! @}

	//! @name Intermediate
	//! @{

	template<typename TBuffer>
	void flush_buffer(size_t number, TBuffer const & ext_buffer);

	DataSet dataset() const;

	//! @}

	//! @name Advanced
	//! @{

	bool is_changed() const
	{
		return engine_type::properties.is_changed() || is_changed_;
	}

	DataSet dump_cache() const;

	void clear_cache();

	void inc_step_counter(size_t num_of_steps)
	{
		step_counter_ += num_of_steps;
	}

	template<typename TFun>
	void foreach(TFun const & fun);

	std::vector<Point_s> buffer;
	std::shared_ptr<Point_s> data;

	//! @}
private:

	bool is_changed_ = false;

	size_t step_counter_ = 0;

	size_t cache_depth_ = 0;

	size_t number_of_points_ = 0;

	CHECK_MEMBER_VALUE(memory_length,0);

	static constexpr size_t memory_length = check_member_value_memory_length<engine_type>::value;

	static constexpr bool is_markov_chain = (check_member_value_memory_length<engine_type>::value==0);

	HAS_MEMBER_FUNCTION(next_timestep);

	template<typename TPIterator ,typename ...Args>
	inline auto next_timestep_selector_(TPIterator p, Real time, Args && ... args)const
	->typename std::enable_if<
	has_member_function_next_timestep<Engine,TPIterator, Real, Args...>::value,void>::type
	{
		engine_type::next_timestep(p, time, std::forward<Args>(args)...);
	}

	template< typename TPIterator ,typename ...Args>
	inline auto next_timestep_selector_( TPIterator p, Real time, Args && ...args)const
	->typename std::enable_if<
	( !has_member_function_next_timestep<Engine,TPIterator, Real, Args...>::value) &&
	( has_member_function_next_timestep<Engine, TPIterator, Args...>::value)
	,void>::type
	{
		engine_type::next_timestep(p, std::forward<Args>(args)...);
	}

	template<typename TPIterator , typename ...Args>
	inline auto next_timestep_selector_( TPIterator p,Real dt,Real time, Args && ...args)const
	->typename std::enable_if<
	( !has_member_function_next_timestep<Engine, TPIterator,Real, Real, Args...>::value) &&
	( !has_member_function_next_timestep<Engine, TPIterator,Real, Args...>::value)
	,void>::type
	{
		RUNTIME_ERROR("Wrong Way");
	}

};

template<typename Engine>
template<typename ... Others>
ProbeParticle<Engine>::ProbeParticle(Others && ...others)
{
}

template<typename Engine>
ProbeParticle<Engine>::~ProbeParticle()
{
}

template<typename Engine>
template<typename TDict, typename ...Others>
void ProbeParticle<Engine>::load(TDict const & dict, Others && ...others)
{

	if (dict["URL"])
	{
		UNIMPLEMENTED2(" read particle from file");
	}
	else
	{
		engine_type::load(dict, std::forward<Others>(others)...);
	}

}
template<typename Engine>
template<typename TBuffer>
void ProbeParticle<Engine>::flush_buffer(size_t num, TBuffer const & ext_buffer)
{
	engine_type::properties("CacheLength").as(&cache_depth_);

	number_of_points_ = num / (memory_length + 1);

	data = sp_make_shared_array<Point_s>(number_of_points_ * cache_depth_);

	//  move data from buffer_ to data_
	parallel_foreach(make_seq_range(0UL, number_of_points_),

	[&](size_t s)
	{
		Point_s * p=data.get();

		for (int i = 0; i <= memory_length; ++i)
		{
			p[s*(cache_depth_+1)+i]=ext_buffer[s*memory_length+i];
		}

	});

}
template<typename Engine>
bool ProbeParticle<Engine>::update()
{
	if (!is_changed())
	{
		return true;
	}

	engine_type::update_properties();

	engine_type::update();

	flush_buffer(buffer.size(), buffer);

	buffer.clear();

	is_changed_ = false;

	return true;

}
template<typename Engine>
void ProbeParticle<Engine>::sync()
{
}

template<typename Engine>
DataSet ProbeParticle<Engine>::dataset() const
{
	size_t dims[2] = { number_of_points_, memory_length + 1 };

	return std::move(make_dataset(data + (

	number_of_points_ * (step_counter_ - memory_length)

	), 1, dims, properties()));
}
template<typename Engine>
DataSet ProbeParticle<Engine>::dump_cache() const
{
	size_t dims[2] = { number_of_points_, step_counter_ };

	return std::move(make_dataset(data, 1, dims, properties()));
}
template<typename Engine>
template<typename TFun>
void ProbeParticle<Engine>::foreach(TFun const& fun)
{

	parallel_foreach(make_seq_range(0UL, number_of_points_), [&](size_t s)
	{
		fun(data.get() + s*(cache_depth_+1));
	});

}

template<typename Engine>
template<typename ... Args>
void ProbeParticle<Engine>::next_timestep(Args && ...args)
{

	parallel_foreach(make_seq_range(0UL, number_of_points_),

	[&](size_t s)
	{
		engine_type::next_timestep(data.get()
				+ s*(cache_depth_+1)+step_counter_
				,std::forward<Args>(args)...);
	});

	inc_step_counter(1);
}

template<typename Engine>
template<typename ... Args>
void ProbeParticle<Engine>::next_n_timesteps(size_t num_of_steps, Real t0,
		Real dt, Args && ...args)
{

	if ((num_of_steps + step_counter_) > cache_depth_)
	{
		size_t n0 = cache_depth_ - step_counter_;
		size_t n1 = num_of_steps + step_counter_ - cache_depth_;

		Real t1 = t0 + n0 * dt;

		next_n_timesteps(n0, t0, dt, std::forward<Args>(args)...);

		next_n_timesteps(n1, t1, dt, std::forward<Args>(args)...);

	}
	else
	{

		for_each([&](Point_s * p)
		{
			for (int i = 0; i < num_of_steps; ++i)
			{
				next_timestep_selector_(p, t0, dt, std::forward<Args>(args)...);
				++p;
				t0 += dt;
			}
		});

		inc_step_counter(num_of_steps);
	}

}

template<typename Engine>
void ProbeParticle<Engine>::clear_cache()
{

	if (step_counter_ >= cache_depth_)
	{
		if (cache_depth_ > 0)
		{
			foreach([&](Point_s * p)
			{
				for (int i = 0; i <= memory_length; ++i)
				{
					p[i]=p[step_counter_-memory_length+i];
				}
			});
		}
		step_counter_ = memory_length;
	}
}
template<typename Engine, typename ...Others>
auto make_probe_particle(Others && ... others)
DECL_RET_TYPE((std::make_shared<ProbeParticle<Engine>>(
						std::forward<Others>(others)...)))

template<typename Engine>
auto make_probe_particle()
DECL_RET_TYPE((std::make_shared< ProbeParticle<Engine>>( )))
}  // namespace simpla

#endif /* CORE_PARTICLE_PROBE_PARTICLE_H_ */
