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

#include "../physics/physical_object.h"
#include "../utilities/log.h"
#include "../utilities/primitives.h"
#include "../utilities/sp_iterator_sequence.h"
#include "../data_structure/data_set.h"
#include "../parallel/parallel.h"
#include "particle.h"
namespace simpla
{
namespace _impl
{
struct IsProbeParticle;
}  // namespace _impl

template<typename ...> struct Particle;
/**
 *  @brief Particle<IsTracable> is a container of particle trajectory.
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
 */

/***
 *  @brief Particle
 *
 *  Require:
 *    engine_type::next_timestep(Point_s * p, others...);
 *
 *  - if  engine_type::memory_length  is not defined
 *     p point the "Particle" at current time step
 *
 *  - if engine_type::memory_length = m
 *    p-1,p-2,... , p-m are valid and point to "Particles" at previous m steps
 *
 */
template<typename Engine, typename TDomain>
struct Particle<Engine, TDomain, _impl::IsProbeParticle> : public PhysicalObject,
		public Engine
{
	typedef PhysicalObject base_type;

	typedef TDomain domain_type;

	typedef Engine engine_type;

	typedef std::vector<typename Engine::Point_s> container_type;

	typedef Particle<engine_type, domain_type, _impl::IsProbeParticle> this_type;

	typedef typename engine_type::Point_s Point_s;

	//***************************************************************************************************
	// Constructor
	template<typename ...Others>
	Particle(Others && ...);	// Constructor

	// Destroy
	~Particle();

	using engine_type::properties;

//	Properties const & properties(std::string const & name = "") const
//	{
//		return engine_type::properties(name);
//	}
//
//	Properties & properties(std::string const & name = "")
//	{
//		return engine_type::properties(name);
//	}

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

	template<typename TFun, typename ...Args>
	void foreach(TFun const & fun, Args && ...args);

	std::ostream& print(std::ostream & os) const
	{
		engine_type::print(os);
		return os;
	}

	bool update();

	DataSet dataset() const;

private:

	std::shared_ptr<Point_s> data_;

	size_t step_counter_ = 0;
	size_t cache_depth_ = 0;

	size_t number_of_particles_ = 0;

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
//#error UNSPORTED next_timestep
//#warning UNSPORTED next_timestep
		RUNTIME_ERROR("Wrong Way");
	}

};

template<typename Engine, typename TDomain>
template<typename ... Others>
Particle<Engine, TDomain, _impl::IsProbeParticle>::Particle(Others && ...others)
{
}

template<typename Engine, typename TDomain>
Particle<Engine, TDomain, _impl::IsProbeParticle>::~Particle()
{
}

template<typename Engine, typename TDomain>
template<typename TDict, typename ...Others>
void Particle<Engine, TDomain, _impl::IsProbeParticle>::load(TDict const & dict,
		Others && ...others)
{

	if (dict["URL"])
	{
		UNIMPLEMENTED2(" read particle from file");

		// TODO read particle from file
	}
	else
	{
		engine_type::load(dict, std::forward<Others>(others)...);
	}

}

template<typename Engine, typename TDomain>
bool Particle<Engine, TDomain, _impl::IsProbeParticle>::update()
{
	engine_type::update_properties();
	engine_type::update();

	return true;
}

template<typename Engine, typename TDomain>
DataSet Particle<Engine, TDomain, _impl::IsProbeParticle>::dataset() const
{
	size_t dims[2] = { number_of_particles_, cache_depth_ };

	return std::move(
			make_dataset(container_type::data(), 1, dims, properties()));
}

template<typename Engine, typename TDomain>
template<typename TFun, typename ... Args>
void Particle<Engine, TDomain, _impl::IsProbeParticle>::foreach(TFun const& fun,
		Args && ...args)
{
	for (auto & item : *this)
	{
		auto p = &item;

		fun(&item, std::forward<Args>(args)...);
	}

//	parallel_foreach(make_seq_range(0UL, num_of_points_),
//
//	[&](size_t n)
//	{
//		Point_s * p = data_.get() + n;
//		next_timestep_(function_selector, p ,dt, std::forward<Args>(args)...);
//	});

//	if (cache_length_ == 0)
//	{
//
//		parallel_foreach(make_seq_range(0UL, num_of_points_),
//
//		[&](size_t n)
//		{
//			Point_s * p = data_.get() + n*cache_length_;
//
//			engine_type::next_timestep(p , std::forward<Args>(args)...);
//
//		}
//
//		);
//
//	}
//	else if (memory_length_ + num_of_steps > cache_length_)
//	{
//		size_t a = cache_length_ - memory_length_;
//		size_t b = num_of_steps - a;
//
//		next_n_timesteps(a, std::forward<Args>(args)...);
//	}
//	else
//	{
//
//		if (is_markov_chain)
//		{
//
//			parallel_foreach(make_seq_range(0UL, num_of_points_),
//
//			[&](size_t n)
//			{
//
//				Point_s * p = data_.get() + n*cache_length_+memory_length_;
//
//				for (size_t s = 0; s < num_of_steps; ++s)
//				{
//					engine_type::next_timestep(p, std::forward<Args>(args)...);
//					*(p +1)=*p;
//					++p;
//				}
//
//			}
//
//			);
//		}
//		else
//		{
//
//			parallel_foreach(make_seq_range(0UL, num_of_points_),
//
//			[&](size_t n)
//			{
//
//				Point_s * p = data_.get() + n*cache_length_+memory_length_;
//
//				for (size_t s = 0; s < num_of_steps; ++s)
//				{
//					engine_type::next_timestep(p, std::forward<Args>(args)...);
//					++p;
//				}
//
//			}
//
//			);
//		}
//
//		memory_length_ += num_of_steps;
//
//		if (memory_length_ == cache_length_)
//		{
//
//			parallel_foreach(make_seq_range(0UL, num_of_points_),
//
//			[&](size_t n)
//			{
//
//				Point_s * p0 = (data_.get() + n*cache_length_);
//
//				for (size_t s = 0; s < min_memory_length; ++s)
//				{
//					*(p0+s)=*(p0+cache_length_-min_memory_length+s);
//				}
//
//			}
//
//			);
//		}
//	}

}

template<typename Engine, typename TDomain>
template<typename ... Args>
void Particle<Engine, TDomain, _impl::IsProbeParticle>::next_timestep(
		Args && ...args)
{
	foreach([&](Point_s * p)
	{
		engine_type::next_timestep(p,std::forward<Args>(args)...);
	});

	++step_counter_;
}

template<typename Engine, typename TDomain>
template<typename ... Args>
void Particle<Engine, TDomain, _impl::IsProbeParticle>::next_n_timesteps(
		size_t num_of_steps, Real t0, Real dt, Args && ...args)
{
	foreach([&](Point_s * p)
	{
		for (int n = 0; n < num_of_steps; ++n)
		{
			next_timestep_selector_(p,t0,dt,std::forward<Args>(args)...);
			++p;
			t0+=dt;
		}

	});

	step_counter_ += num_of_steps;
}

template<typename Engine>
using ProbeParticle=Particle< Engine, std::nullptr_t, _impl::IsProbeParticle>;

template<typename Engine, typename ...Others>
auto make_probe_particle(Others && ... others)
DECL_RET_TYPE((std::make_shared<ProbeParticle<Engine>>(
						std::forward<Others>(others)...)))

template<typename Engine>
auto make_probe_particle()
DECL_RET_TYPE((std::make_shared< ProbeParticle<Engine>>( )))
}  // namespace simpla

#endif /* CORE_PARTICLE_PROBE_PARTICLE_H_ */
