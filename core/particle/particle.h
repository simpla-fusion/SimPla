/*
 * particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "particle_engine.h"
#include "kinetic_particle.h"
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

//
///** \defgroup  Particle Particle
// *
// *  \brief Particle  particle concept
// */
//
//template<typename TM, typename Engine, typename Policy>
//class Particle: public Engine
//{
//
//public:
//	static constexpr unsigned int IForm = VERTEX;
//
//	typedef TM mesh_type;
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
