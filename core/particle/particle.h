/*
 * particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "particle_engine.h"
#include "load_particle.h"
#include "../physics/physical_object.h"
#include "../parallel/parallel.h"
#include "../utilities/sp_iterator_sequence.h"
namespace simpla
{

/** \defgroup  Particle Particle
 *
 *  \brief Particle  particle concept
 */

template<typename ...>struct Particle;

template<typename ...T>
std::ostream& operator<<(std::ostream & os, Particle<T...> const &p)
{
	p.print(os);

	return os;
}
/***
 *  - if is  Markov chain
 *    p_n = f(p_{n-1})
 *    address  &p_n = &p_{n-1}
 *
 *  - if is  not Markov chain
 *	  p_n = f(p_{n-1},p_{n-2},p_{n-3},...,p_{n-m})
 *	  address  &p_n = &p_{n-1} + 1
 *
 */
template<typename TDomain, typename Engine>
struct Particle<TDomain, Engine> : public PhysicalObject, public Engine
{
	typedef PhysicalObject base_type;

	typedef TDomain domain_type;

	typedef Engine engine_type;

	typedef Particle<domain_type, engine_type> this_type;

	typedef typename engine_type::Point_s Point_s;

private:

	CHECK_VALUE(is_markov_chain,true);

	CHECK_VALUE(memory_length,(check_is_markov_chain<
					engine_type>::value?0:1));
public:

	static constexpr size_t memory_length = check_memory_length<engine_type>::value;

	static constexpr bool is_markov_chain = memory_length==0;

	//***************************************************************************************************
	// Constructor
	template<typename ...Others>
	Particle(domain_type const &, Others && ...);// Constructor

	// Destroy
	~Particle();

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others)
	{
		return engine_type::load(dict,std::forward<Others>(others)...);
	}

	static std::string get_type_as_string_staic()
	{
		return engine_type::get_type_as_string();
	}

	template<typename ...Args>
	void next_n_timesteps(size_t num_of_steps, Args && ...args);

	template<typename ...Args>
	void next_timestep( Args && ...args)
	{
		next_n_steps(1,std::forward<Args>(args)...);
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

	bool update()
	{
		engine_type::update();

		properties("Cache Length").as(&chain_length_);

		return true;
	}

	DataSet dataset()const
	{
		return DataSet();
	}

private:

	domain_type domain_;

	std::shared_ptr<Point_s> data_;

	size_t clock_ = 0;

	size_t num_of_points_ = 1024;

	size_t chain_length_ = 1;

};

template<typename Engine, typename TDomain>
template<typename ... Others>
Particle<Engine, TDomain>::Particle(domain_type const &d, Others && ...others) :
		domain_(d)
{
	engine_type::load(std::forward<Others>(others)...);
}

template<typename Engine, typename TDomain>
Particle<Engine, TDomain>::~Particle()
{
}

template<typename Engine, typename TDomain>
template<typename ... Args>
void Particle<Engine, TDomain>::next_n_timesteps(size_t num_of_steps,
		Args && ...args)
{
	Point_s * head_ = data_.get() + clock_;

	parallel_for(make_seq_range(0UL, num_of_steps),

	[&](size_t n)
	{

		Point_s * p0 = head_ + n;

		for (size_t s = 0; s < num_of_steps; ++s)
		{
			engine_type::next_timestep(p0 + s, std::forward<Args>(args)...);
		}

	}

	);

	clock_ += num_of_steps;

//	head_ %= cache_length_ * num_of_points_;

}
template<typename Engine, typename TDomain, typename ...Others>
auto make_particle(TDomain const & d, Others && ... others)
DECL_RET_TYPE((std::make_shared<Particle<TDomain,Engine>>(
						d,std::forward<Others>(others)...)))
}
// namespace simpla

#endif /* PARTICLE_H_ */
