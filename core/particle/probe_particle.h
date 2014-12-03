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
template<typename TDomain, typename Engine>
struct Particle<TDomain, Engine, _impl::IsProbeParticle> : public PhysicalObject,
		public Engine
{
	typedef PhysicalObject base_type;

	typedef TDomain domain_type;

	typedef Engine engine_type;

	typedef Particle<domain_type, engine_type, _impl::IsProbeParticle> this_type;

	typedef typename engine_type::Point_s Point_s;

	//***************************************************************************************************
	// Constructor
	template<typename ...Others>
	Particle(domain_type const &, Others && ...);	// Constructor

	// Destroy
	~Particle();

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others);

	static std::string get_type_as_string_staic()
	{
		return engine_type::get_type_as_string();
	}

	template<typename ...Args>
	void next_n_timesteps(size_t num_of_steps, Args && ...args);

	template<typename ...Args>
	void next_timestep(Args && ...args)
	{
		next_n_steps(1, std::forward<Args>(args)...);
	}

	Properties const & properties(std::string const & key = "") const
	{
		return engine_type::properties(key);
	}

	Properties & properties(std::string const & key = "")
	{
		return engine_type::properties(key);
	}

	std::string get_type_as_string() const
	{
		return get_type_as_string_staic();
	}

	std::ostream& print(std::ostream & os) const
	{
		engine_type::print(os);
		base_type::print(os);
		return os;
	}

	bool update();

	void sync();

	DataSet dataset() const;

private:

	domain_type domain_;

	std::shared_ptr<Point_s> data_;

	size_t num_of_points_ = 0;

	size_t memory_length_ = 0;

	size_t cache_length_ = 0;

	CHECK_VALUE(is_markov_chain,true);

	CHECK_VALUE(memory_length,(check_is_markov_chain<
					engine_type>::value?0:1));

	static constexpr size_t min_memory_length = check_memory_length<engine_type>::value;

public:
	static constexpr bool is_markov_chain = min_memory_length==0;

};

template<typename Engine, typename TDomain>
template<typename ... Others>
Particle<Engine, TDomain, _impl::IsProbeParticle>::Particle(
		domain_type const &d, Others && ...others) :
		domain_(d)
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
	engine_type::update();

	properties("Cache Length").as(&cache_length_);

	return true;
}

template<typename Engine, typename TDomain>
void Particle<Engine, TDomain, _impl::IsProbeParticle>::sync()
{
	UNIMPLEMENTED2(" update ghost ");
}

template<typename Engine, typename TDomain>
DataSet Particle<Engine, TDomain, _impl::IsProbeParticle>::dataset() const
{
	DataSpace ds;
	if (cache_length_ == 0)
	{
		ds = make_dataspace(1, &num_of_points_);
	}
	else
	{
		size_t dims[2] =
		{ num_of_points_, memory_length_ };

		ds = make_dataspace(2, dims);

	}

	return std::move(DataSet(
	{

	properties(),

	data_,

	make_datatype<Point_s>(),

	ds

	}));
}

template<typename Engine, typename TDomain>
template<typename ... Args>
void Particle<Engine, TDomain, _impl::IsProbeParticle>::next_n_timesteps(
		size_t num_of_steps, Args && ...args)
{

	if (cache_length_ == 0)
	{

		parallel_for(make_seq_range(0UL, num_of_points_),

		[&](size_t n)
		{
			Point_s * p = data_.get() + n*cache_length_;

			for (size_t s = 0; s < num_of_steps; ++s)
			{
				engine_type::next_timestep(p , std::forward<Args>(args)...);
			}
		}

		);

	}
	else if (memory_length_ + num_of_steps > cache_length_)
	{
		size_t a = cache_length_ - memory_length_;
		size_t b = num_of_steps - a;

		next_n_timesteps(a, std::forward<Args>(args)...);
		next_n_timesteps(a, std::forward<Args>(args)...);
	}
	else
	{

		if (is_markov_chain)
		{

			parallel_for(make_seq_range(0UL, num_of_points_),

			[&](size_t n)
			{

				Point_s * p = data_.get() + n*cache_length_+memory_length_;

				for (size_t s = 0; s < num_of_steps; ++s)
				{
					engine_type::next_timestep(p, std::forward<Args>(args)...);
					*(p +1)=*p;
					++p;
				}

			}

			);
		}
		else
		{

			parallel_for(make_seq_range(0UL, num_of_points_),

			[&](size_t n)
			{

				Point_s * p = data_.get() + n*cache_length_+memory_length_;

				for (size_t s = 0; s < num_of_steps; ++s)
				{
					engine_type::next_timestep(p, std::forward<Args>(args)...);
					++p;
				}

			}

			);
		}

		memory_length_ += num_of_steps;

		if (memory_length_ == cache_length_)
		{

			parallel_for(make_seq_range(0UL, num_of_points_),

			[&](size_t n)
			{

				Point_s * p0 = (data_.get() + n*cache_length_);

				for (size_t s = 0; s < min_memory_length; ++s)
				{
					*(p0+s)=*(p0+cache_length_-min_memory_length+s);
				}

			}

			);
		}
	}

}
template<typename TDomain, typename Engine>
using ProbeParticle=Particle<TDomain, Engine, _impl::IsProbeParticle>;

template<typename Engine, typename TM, typename ...Others>
auto make_probe_particle(std::shared_ptr<TM> d, Others && ... others)
DECL_RET_TYPE((std::make_shared<ProbeParticle<Domain<TM,VERTEX>,Engine >>(
						d,std::forward<Others>(others)...)))


}  // namespace simpla

#endif /* CORE_PARTICLE_PROBE_PARTICLE_H_ */
