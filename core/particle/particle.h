/*
 * particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "particle_engine.h"
#include "probe_particle.h"
#include "save_particle.h"
#include "load_particle.h"

namespace simpla
{
template<typename ...T>
std::ostream& operator<<(std::ostream & os, Particle<T...> const &p)
{
	p.print(os);

	return os;
}

/** \defgroup  Particle Particle
 *
 *  \brief Particle  particle concept
 */

template<typename ...>struct Particle;

template<typename Engine, typename TDomain>
struct Particle<Engine, TDomain> : public Engine, public std::vector<
		typename Engine::Point_s>
{
	typedef TDomain domain_type;
	typedef Engine engine_type;

	typedef Particle<domain_type, engine_type> this_type;

	typedef std::vector<typename Engine::Point_s> storage_type;

	typedef typename engine_type::Point_s particle_type;

	typedef typename engine_type::scalar_type scalar_type;

	typedef particle_type value_type;

	typedef typename domain_type::iterator iterator;

	typedef typename domain_type::coordinates_type coordinates_type;

	domain_type const & domain_;

	//***************************************************************************************************
	// Constructor
	template<typename ...Others>
	Particle(domain_type const & pmesh, Others && ...);	// Constructor

	// Destructor
	~Particle();

	static std::string get_type_as_string()
	{
		return engine_type::get_type_as_string();
	}

	void load();

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others);

	template<typename ...Args>
	std::string save(std::string const & path, Args &&...) const;

	std::ostream& print(std::ostream & os) const
	{
		engine_type::print(os);
		return os;
	}

	template<typename ...Args>
	void multi_timesteps(size_t step_num, Args && ...args)
	{
		for (size_t s = 0; s < step_num; ++s)
		{
			next_timestep(std::forward<Args>(args)...);
		}
	}

	template<typename ...Args>
	void next_timestep(Real dt, Args && ...);

	template<typename ...Args>
	auto emplace_back(Args && ...args)
	DECL_RET_TYPE((storage_type::emplace_back(particle_type(
									{	args...}))))
};

template<typename Engine, typename TDomain>
template<typename ... Others>
Particle<Engine, TDomain>::Particle(domain_type const & pmesh,
		Others && ...others) :
		domain_(pmesh)
{
	engine_type::load(std::forward<Others>(others)...);
}

template<typename Engine, typename TDomain>
Particle<Engine, TDomain>::~Particle()
{
}

template<typename Engine, typename TDomain>
template<typename ...Args>
std::string Particle<Engine, TDomain>::save(std::string const & path,
		Args && ...args) const
{
	return save(path, *this, std::forward<Args>(args)...);
}

template<typename Engine, typename TDomain>
template<typename ... Args>
void Particle<Engine, TDomain>::next_timestep(Real dt, Args && ... args)
{

	LOGGER << "Push probe particles   [ " << get_type_as_string() << "]"
			<< std::endl;

	auto p = &*(this->begin());
	for (auto & p : *this)
	{

		engine_type::next_timestep(&p, dt, std::forward<Args>(args)...
		/*,	engine_type::mass, engine_type::charge,
		 engine_type::temperature*/);

	}

	LOGGER << DONE;
}

//
//template<typename TDomain, typename Engine, typename Policy>
//class Particle: public Engine
//{
//
//public:
//	static constexpr unsigned int IForm = VERTEX;
//
//	typedef TDomain mesh_type;
//	typedef Engine engine_type;
//
//	typedef Particle<mesh_type, engine_type, Policy> this_type;
//
//
//	mesh_type const & mesh;
//
//	template<typename ...Others>
//	Particle(mesh_type const & pmesh, Others && ...); 	//! Constructor
//
//	~Particle();	//! Destructor
//
//	template<typename ...Others>
//	void load(Others && ...others); //! load / configure
//
//	static std::string get_type_as_string(); //! get type name or id
//
//	std::string save(std::string const & path) const; //! save particle to io
//
//	std::ostream& print(std::ostream & os) const; //! print particle description
//
//	template<typename ...Args> void next_timestep(Args && ...); //! push particle to next time step
//
//	template<typename TJ> void ScatterJ(TJ * pJ) const; //! accumulate particle current density to *pJ
//
//	template<typename TJ> void ScatterRho(TJ * prho) const; //! accumulate particle charge density to *prho
//
//};
//
}
// namespace simpla

#endif /* PARTICLE_H_ */
