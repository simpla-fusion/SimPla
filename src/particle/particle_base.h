/*
 * particle_base.h
 *
 *  Created on: 2014年1月16日
 *      Author: salmon
 */

#ifndef PARTICLE_BASE_H_
#define PARTICLE_BASE_H_

#include <bits/shared_ptr.h>
#include <cstddef>
#include <iostream>
#include <string>

//#include "../fetl/fetl.h"
#include "../utilities/log.h"
#include "../utilities/lua_state.h"
#include "particle.h"

namespace simpla
{
class LuaObject;
template<typename TM>
class ParticleBase
{

public:
	typedef TM mesh_type;

	DEFINE_FIELDS (mesh_type)

	enum
	{
		REFELECT, ABSORB
	};

	ParticleBase()
			: isSorted_(false), clock_(0)
	{
	}
	virtual ~ParticleBase()
	{
	}

	virtual void Update()
	{
	}

	virtual void DumpData(std::string const &path = "") const
	{

	}

	virtual void Deserialize(LuaObject const &cfg)
	{
	}

	virtual std::string GetTypeAsString() const
	{
		return "unknown";
	}

	virtual std::ostream & Serialize(std::ostream & os) const
	{
		return os;
	}

	bool IsSorted() const
	{
		return isSorted_;
	}

	Real GetClock() const
	{
		return clock_;
	}

	void SetClock(Real clock)
	{
		clock_ = clock;
	}

//interface
	virtual void NextTimeStep(double dt, Form<1> const &E, Form<2> const &B)
	{
		isSorted_ = false;
		clock_ += dt;
	}
	virtual void NextTimeStep(double dt, VectorForm<0> const &E, VectorForm<0> const &B)
	{
		isSorted_ = false;
		clock_ += dt;
	}

	virtual void Boundary(int flag, typename mesh_type::tag_type in, typename mesh_type::tag_type out, double dt,
	        Form<1> const &E, Form<2> const &B)
	{
		UNIMPLEMENT;
	}
	virtual void Boundary(int flag, typename mesh_type::tag_type in, typename mesh_type::tag_type out, double dt,
	        VectorForm<0> const &E, VectorForm<0> const &B)
	{
		UNIMPLEMENT;
	}
	virtual void Collide(Real dt, ParticleBase *)
	{
		UNIMPLEMENT;
	}

	virtual void Collect(Form<0> * n, Form<1> const &E, Form<2> const &B) const
	{
		UNIMPLEMENT;
	}
	virtual void Collect(Form<1> * J, Form<1> const &E, Form<2> const &B) const
	{
		UNIMPLEMENT;
	}
	virtual void Collect(Form<2> * J, Form<1> const &E, Form<2> const &B) const
	{
		UNIMPLEMENT;
	}
	virtual void Collect(VectorForm<0> * J, Form<1> const &E, Form<2> const &B) const
	{
		UNIMPLEMENT;
	}
	virtual void Collect(VectorForm<1> * P, Form<1> const &E, Form<2> const &B) const
	{
		UNIMPLEMENT;
	}
	virtual void Collect(VectorForm<2> * P, Form<1> const &E, Form<2> const &B) const
	{
		UNIMPLEMENT;
	}
	virtual void Collect(TensorForm<0> * P, Form<1> const &E, Form<2> const &B) const
	{
		UNIMPLEMENT;
	}
	virtual void Sort()
	{
		isSorted_ = true;
	}

	void SetName(std::string const & name)
	{
		name_ = name;
	}
	std::string const &GetName() const
	{
		return name_;
	}

private:
	bool isSorted_;
	Real clock_;
	std::string name_;

};

//*******************************************************************************************************
template<typename TM>
struct PICEngineBase
{

protected:
	Real m_, q_;
public:
	typedef TM mesh_type;

public:

	mesh_type const &mesh;

	PICEngineBase(mesh_type const &pmesh)
			: mesh(pmesh), m_(1.0), q_(1.0)
	{
	}
	virtual ~PICEngineBase()
	{
	}

	virtual std::string GetTypeAsString() const
	{
		return "unknown";
	}

	virtual size_t GetAffectedRegion() const
	{
		return 2;
	}

	inline Real GetMass() const
	{
		return m_;
	}

	inline Real GetCharge() const
	{
		return q_;
	}

	inline void SetMass(Real m)
	{
		m_ = m;
	}

	inline void SetCharge(Real q)
	{
		q_ = q;
	}

	virtual void Update()
	{
	}

	virtual void Deserialize(LuaObject const &vm)
	{
		m_ = vm["Mass"].as<Real>();
		q_ = vm["Charge"].as<Real>();
	}

	virtual std::ostream & Serialize(std::ostream & os) const
	{
		os

		<< "Mass = " << m_ << " , "

		<< "Charge = " << q_;

		return os;
	}

};
//*******************************************************************************************************

template<typename > class Particle;
template<typename TParticleEngine>
std::shared_ptr<ParticleBase<typename TParticleEngine::mesh_type> > CreateParticle(
        typename TParticleEngine::mesh_type const & mesh)
{

	typedef Particle<TParticleEngine> particle_type;
	typedef typename TParticleEngine::mesh_type mesh_type;

	return std::dynamic_pointer_cast<ParticleBase<mesh_type> >(
	        std::shared_ptr<ParticleBase<mesh_type> >(new particle_type(mesh)));
}
}  // namespace simpla

#endif /* PARTICLE_BASE_H_ */
