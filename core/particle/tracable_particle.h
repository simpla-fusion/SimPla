/*
 * probe_particle.h
 *
 *  Created on: 2014年7月12日
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_TRACABLE_PARTICLE_H_
#define CORE_PARTICLE_TRACABLE_PARTICLE_H_

#include <iostream>
#include <memory>
#include <string>

#include "../physics/physical_object.h"
#include "../utilities/log.h"
#include "../utilities/primitives.h"
#include "../data_structure/data_set.h"
namespace simpla
{

struct IsTracable;
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

template<typename Engine, typename TDomain>
struct Particle<Engine, TDomain> : public Engine, public PhysicalObject
{
	typedef TDomain domain_type;

	typedef Engine engine_type;

	typedef Particle<domain_type, engine_type> this_type;

	typedef typename Engine::Point_s Point_s;

	typedef Point_s * iterator;

	typedef typename domain_type::coordinates_type coordinates_type;

	//****************************************************************
	// Constructor
	template<typename ...Others>
	Particle(std::shared_ptr<domain_type>& pdomain, Others && ...);

	// Destroy
	~Particle();

	static std::string get_type_as_string_static()
	{
		return engine_type::get_type_as_string();
	}
	std::string get_type_as_string() const
	{
		return get_type_as_string_static();
	}

	template<typename TDict>
	void load(TDict const & dict)
	{
		engine_type::load(dict);
	}

	bool update();

//	void dataset(DataSet);

	DataSet dataset() const;

	void swap(this_type &);

	template<typename OS>
	OS& print(OS& os) const;

	template<typename ...Args>
	void next_n_timestep(size_t step_num, Real dt, Args && ...);

	template<typename ...Args>
	void next_timestep(Real dt, Args && ...);

	void push_back(Point_s const &);

	template<typename TI>
	void push_back(TI const & b, TI const & e);

	template<typename ...Args>
	void emplace_back(Args && ...args);

private:
	std::shared_ptr<domain_type> domain_;

	std::shared_ptr<Point_s> data_;

	size_t num_of_points_ = 1024;
	size_t cache_length_ = 1;
	bool is_cached_ = false;
	size_t cache_depth_ = 10;

};

template<typename Engine, typename TDomain>
template<typename OS>
OS& Particle<Engine, TDomain>::print(OS & os) const
{
	engine_type::print(os);
	return os;
}

template<typename Engine, typename TDomain>
template<typename ... Others>
Particle<Engine, TDomain>::Particle(std::shared_ptr<TDomain> & pdomain,
		Others && ...others) :
		domain_(pdomain->shared_from_this()), engine_type(
				std::forward<Others>(others)...)
{
//	engine_type::load(std::forward<Others>(others)...);
}

template<typename Engine, typename TDomain>
Particle<Engine, TDomain>::~Particle()
{
}
template<typename Engine, typename TDomain>
bool Particle<Engine, TDomain>::update()
{
	return true;
}
template<typename Engine, typename TDomain>
DataSet Particle<Engine, TDomain>::dataset() const
{
	DataSet res;
	return (res);
}

template<typename Engine, typename TDomain>
template<typename ... Args>
void Particle<Engine, TDomain>::next_n_timestep(size_t step, Real dt,
		Args && ... args)
{

	LOGGER << "Push probe particles   [ " << get_type_as_string() << "]"
			<< std::endl;

//	auto p = &*(this->begin());
//	for (auto & p : *this)
//	{
//		for (int s = 0; s < step; ++s)
//		{
//			engine_type::next_timestep(p + s, dt, std::forward<Args>(args)...);
//		}
//	}

	LOGGER << DONE;
}

template<typename Engine, typename TDomain>
template<typename ... Args>
void Particle<Engine, TDomain>::next_timestep(Real dt, Args && ... args)
{

	LOGGER << "Push probe particles   [ " << get_type_as_string() << "]"
			<< std::endl;

//	auto p = &*(this->begin());
//	for (auto & p : *this)
//	{
//			engine_type::next_timestep(p + s, dt, std::forward<Args>(args)...);
//	}

	LOGGER << DONE;
}

}  // namespace simpla

#endif /* CORE_PARTICLE_TRACABLE_PARTICLE_H_ */
