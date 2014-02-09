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


}  // namespace simpla

#endif /* PARTICLE_BASE_H_ */
