/**
 *  kinetic_particle.h
 *
 * \date    2014年9月1日  下午2:25:26 
 * \author salmon
 */

#ifndef CORE_PARTICLE_KINETIC_PARTICLE_H_
#define CORE_PARTICLE_KINETIC_PARTICLE_H_

#include <iostream>
#include <memory>
#include <string>

#include "../physics/physical_object.h"
#include "../utilities/log.h"
#include "../utilities/primitives.h"
#include "../data_structure/container_pool.h"
namespace simpla
{

/**
 *  @ingroup Particle
 *
 *  @brief KineticParticle is a container of untracable particle  .
 *  -  KineticParticle is a container of untracable particle  .
 *  -  KineticParticle can be sorted;
 *  -  KineticParticle is an unordered container;
 */
template<typename TDomain, typename Engine>
struct KineticParticle: public Engine, public PhysicalObject
{

	typedef PhysicalObject base_type;

	typedef TDomain domain_type;

	typedef Engine engine_type;

	typedef KineticParticle<domain_type, engine_type> this_type;

	typedef typename engine_type::Point_s Point_s;

	typedef typename domain_type::index_type mid_type; // id of mesh point

	//***************************************************************************************************
	// Constructor
	template<typename ...Others>
	KineticParticle(domain_type const &, Others && ...);	// Constructor

	// Destroy
	~KineticParticle();

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
		next_n_timesteps(1, std::forward<Args>(args)...);
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

	ContainerPool<mid_type, typename engine_type::Point_s> data_;

	std::function<mid_type(Point_s const &)> hash_fun_;

};

template<typename TDomain, typename Engine>
template<typename ... Others>
KineticParticle<TDomain, Engine>::KineticParticle(domain_type const & pdomain,
		Others && ...others) :
		domain_(pdomain)
{

}

template<typename TDomain, typename Engine>
KineticParticle<TDomain, Engine>::~KineticParticle()
{
}
template<typename TDomain, typename Engine>
template<typename TDict, typename ...Others>
void KineticParticle<TDomain, Engine>::load(TDict const & dict,
		Others && ...others)
{
	engine_type::load(dict, std::forward<Others>(others)...);
//	data_.load(dict, std::forward<Others>(others)...);
}

template<typename TDomain, typename Engine>
bool KineticParticle<TDomain, Engine>::update()
{

	hash_fun_ = [& ](Point_s const & p)->mid_type
	{
		return std::get<0>(domain_.manifold()->coordinates_global_to_local(
						std::get<0>(engine_type::pull_back(p))));
	};

	return true;
}

template<typename TDomain, typename Engine>
void KineticParticle<TDomain, Engine>::sync()
{
//	data_.sort(hash_fun_);
//	update_ghost(*this);
}

template<typename TDomain, typename Engine>
DataSet KineticParticle<TDomain, Engine>::dataset() const
{
	return DataSet();
}
template<typename TDomain, typename Engine>
template<typename ...Args>
void KineticParticle<TDomain, Engine>::next_n_timesteps(size_t num_of_steps,
		Args && ...args)
{

//	parallel_for(domain_,
//
//	[&](mid_type n)
//	{
//		for(auto & p :data_[n])
//		{
//			for (size_t s = 0; s < num_of_steps; ++s)
//			{
//				engine_type::next_timestep(&p , std::forward<Args>(args)...);
//			}
//		}
//	}
//
//	);

}

//template<typename Engine, typename TDomain, typename ...Others>
//auto make_kinetic_particle(TDomain const & d, Others && ... others)
//DECL_RET_TYPE((std::make_shared<KineticParticle<TDomain,Engine >>(
//						d,std::forward<Others>(others)...)))

template<typename Engine, typename TDomain, typename ...Others>
auto make_kinetic_particle(std::shared_ptr<TDomain> d,
		Others && ... others)
				DECL_RET_TYPE((std::make_shared<KineticParticle<Domain<TDomain,VERTEX>,Engine >>(
										Domain<TDomain,VERTEX>(d),std::forward<Others>(others)...)))

}  // namespace simpla

#endif /* CORE_PARTICLE_KINETIC_PARTICLE_H_ */
