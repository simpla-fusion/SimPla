/*
 * untracable_particle.h
 *
 *  Created on: 2014年11月18日
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_UNTRACABLE_PARTICLE_H_
#define CORE_PARTICLE_UNTRACABLE_PARTICLE_H_
#include <iostream>
#include <memory>
#include <string>

#include "../physics/physical_object.h"
#include "../utilities/log.h"
#include "../utilities/primitives.h"

namespace simpla
{
struct IsUntracable;
template<typename ...> struct _Particle;

/**
 *  @brief Particle<IsUntracable> is a container of untracable particle  .
 *
 *  -  Particle<IsUntracable> can be sorted;
 *  -  Particle<IsUntracable> is an unordered container;
 */

template<typename TDomain, typename Engine>
struct _Particle<TDomain, Engine, IsUntracable> : public Engine,
		public PhysicalObject
{

	typedef TDomain domain_type;
	typedef Engine engine_type;
	typedef _Particle<domain_type, engine_type, IsUntracable> this_type;

	typedef typename domain_type::scalar_type scalar_type;

	typedef typename engine_type::Point_s particle_type;

	typedef typename domain_type::compact_index_type mid_type; // id of mesh point

	typedef ContainerPool<mid_type, typename engine_type::Point_s> storage_type;

	storage_type pic_;

	std::function<compact_index_type(typename engine_type::Point_s const &)> hash_fun_;

	domain_type const & domain_;

	template<typename ...Others>
	Particle(domain_type const & pdomain, Others && ...); // Constructor

	~Particle(); // Destructor

	static std::string get_type_as_string()
	{
		return "Kinetic" + engine_type::get_type_as_string();
	}

	domain_type const & domain() const
	{
		return domain_;
	}
	void load()
	{
	}

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others);

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const;

	template<typename ...Args> void next_timestep(Real dt, Args && ...args);

};

template<typename TM, typename Engine>
template<typename ... Others>
_Particle<TM, Engine, IsUntracable>::Particle(domain_type const & pdomain,
		Others && ...others) :
		domain_(pdomain)
{
	load(std::forward<Others>(others)...);
	hash_fun_ = [& ](particle_type const & p)->mid_type
	{
		return std::get<0>(domain_.coordinates_global_to_local(
						std::get<0>(engine_type::pull_back(p))));
	};
}

template<typename TM, typename Engine>
_Particle<TM, Engine, IsUntracable>::~Particle()
{
}
template<typename TM, typename Engine>
template<typename TDict, typename ...Others>
void _Particle<TM, Engine, IsUntracable>::load(TDict const & dict,
		Others && ...others)
{
	engine_type::load(dict, std::forward<Others>(others)...);

	pic_.load(dict, std::forward<Others>(others)...);

}
template<typename TM, typename Engine>
std::string _Particle<TM, Engine, IsUntracable>::save(
		std::string const & path) const
{
//	std::stringstream os;
//
//	GLOBAL_DATA_STREAM.cd(path);
//
//	os << "\n, n =" << simpla::save("rho", rho);
//
//	os << "\n, J_ =" << simpla::save("J_", J);
//
//	if (properties["DumpParticle"].template as<bool>(false))
//	{
//		os << "\n, particles = " << save(pic_, "particles");
//	}
	return simpla::save(path, *this);
}
template<typename TM, typename Engine>
std::ostream& _Particle<TM, Engine, IsUntracable>::print(std::ostream & os) const
{
	engine_type::print(os);
}

template<typename TM, typename Engine>
template<typename ...Args>
void _Particle<TM, Engine, IsUntracable>::next_timestep(Real dt, Args && ...args)
{

	LOGGER << "Push particles to  next step [ "
			<< engine_type::get_type_as_string() << " ]";

	pic_.sort(hash_fun_);

	pic_.modify(domain_, [&](particle_type & p)
	{
		this->engine_type::next_timestep(&p,dt, std::forward<Args>(args)...);
	});

	pic_.sort(hash_fun_);

	update_ghost(std::forward<Args>(args)...);
}

}  // namespace simpla

#endif /* CORE_PARTICLE_UNTRACABLE_PARTICLE_H_ */
