/*
 * particle_base.h
 *
 *  Created on: 2014年1月16日
 *      Author: salmon
 */

#ifndef PARTICLE_BASE_H_
#define PARTICLE_BASE_H_

#include <sstream>
#include <string>

#include "../fetl/fetl.h"
#include "../io/data_stream.h"
#include "../modeling/select.h"
#include "../modeling/surface.h"
#include "../utilities/visitor.h"
namespace simpla
{

//*******************************************************************************************************
template<typename TM>
struct ParticleBase
{

public:

	typedef TM mesh_type;

	typedef typename mesh_type::scalar_type scalar_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	ParticleBase()
	{
	}

	virtual ~ParticleBase()
	{
	}

	//interface

	virtual std::string Save(std::string const & path) const=0;

	virtual Real GetMass() const=0;

	virtual Real GetCharge() const=0;

	virtual bool EnableImplicit() const = 0;

	virtual std::string GetTypeAsString() const = 0;

	virtual Field<mesh_type, VERTEX, scalar_type> const& n() const=0;
	virtual Field<mesh_type, EDGE, scalar_type> const&J() const=0;
	virtual Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const&Jv() const= 0;

	virtual void NextTimeStepZero(Field<mesh_type, EDGE, scalar_type> const & E,
	        Field<mesh_type, FACE, scalar_type> const & B)
	{
	}
	virtual void NextTimeStepZero(Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
	{
	}
	virtual void NextTimeStepHalf(Field<mesh_type, EDGE, scalar_type> const & E,
	        Field<mesh_type, FACE, scalar_type> const & B)
	{
	}
	virtual void NextTimeStepHalf(Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
	{
	}

};
template<typename TP>
struct ParticleWrap: public ParticleBase<typename TP::mesh_type>
{
	typedef TP particle_type;
	typedef typename particle_type::mesh_type mesh_type;
	typedef typename mesh_type::scalar_type scalar_type;
	typedef ParticleBase<mesh_type> base_type;

	template<typename TDict, typename ...Args>
	ParticleWrap(mesh_type const & mesh, TDict const & dict, Args const & ... args)
			: self_(mesh, dict, std::forward<Args const &>(args)...), dummy_J(mesh), dummy_Jv(mesh)
	{
	}
	~ParticleWrap()
	{
	}
	Real GetMass() const
	{
		return self_.m;
	}

	Real GetCharge() const
	{
		return self_.q;
	}

	bool EnableImplicit() const
	{
		return self_.EnableImplicit;
	}
	std::string GetTypeAsString() const
	{
		return self_.GetTypeAsString();
	}
	Field<mesh_type, VERTEX, scalar_type> const& n() const
	{
		return self_.n;
	}
	Field<mesh_type, EDGE, scalar_type> const& J() const
	{
		return J_(Bool2Type<particle_type::EnableImplicit>());
	}
	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const&Jv() const
	{
		return Jv_(Bool2Type<particle_type::EnableImplicit>());
	}

	void NextTimeStepZero(Field<mesh_type, EDGE, scalar_type> const & E, Field<mesh_type, FACE, scalar_type> const & B)
	{
		NextTimeStepZero_(Bool2Type<particle_type::EnableImplicit>(), E, B);
	}
	void NextTimeStepZero(Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
	{
		NextTimeStepZero_(Bool2Type<particle_type::EnableImplicit>(), E, B);
	}
	void NextTimeStepHalf(Field<mesh_type, EDGE, scalar_type> const & E, Field<mesh_type, FACE, scalar_type> const & B)
	{
		NextTimeStepHalf_(Bool2Type<particle_type::EnableImplicit>(), E, B);
	}
	void NextTimeStepHalf(Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
	{
		NextTimeStepHalf_(Bool2Type<particle_type::EnableImplicit>(), E, B);
	}

	std::string Save(std::string const & path) const
	{
		return self_.Save(path);
	}

private:

//	typename std::enable_if<!particle_type::EnableImplicit, Field<mesh_type, EDGE, scalar_type> const&>::type J_() const
//	{
//		return self_.J;
//	}
//
//	typename std::enable_if<particle_type::EnableImplicit, Field<mesh_type, EDGE, scalar_type> const&>::type J_() const
//	{
//		return dummy_J;
//	}
//
//	typename std::enable_if<particle_type::EnableImplicit, Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const&>::type Jv_() const
//	{
//		return self_.J;
//	}
//
//	typename std::enable_if<!particle_type::EnableImplicit, Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const&>::type Jv_() const
//	{
//		return dummy_Jv;
//	}

	Field<mesh_type, EDGE, scalar_type> const& J_(Bool2Type<false>) const
	{
		return self_.J;
	}

	Field<mesh_type, EDGE, scalar_type> const& J_(Bool2Type<true>) const
	{
		return dummy_J;
	}

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const& Jv_(Bool2Type<true>) const
	{
		return self_.J;
	}

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const& Jv_(Bool2Type<false>) const
	{
		return dummy_Jv;
	}
	void NextTimeStepZero_(Bool2Type<true>, Field<mesh_type, EDGE, scalar_type> const & E,
	        Field<mesh_type, FACE, scalar_type> const & B)
	{
	}
	void NextTimeStepZero_(Bool2Type<false>, Field<mesh_type, EDGE, scalar_type> const & E,
	        Field<mesh_type, FACE, scalar_type> const & B)
	{
		self_.NextTimeStepZero(E, B);
	}
	void NextTimeStepZero_(Bool2Type<true>, Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
	{
		self_.NextTimeStepZero(E, B);
	}
	void NextTimeStepZero_(Bool2Type<false>, Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
	{
	}

	void NextTimeStepHalf_(Bool2Type<false>, Field<mesh_type, EDGE, scalar_type> const & E,
	        Field<mesh_type, FACE, scalar_type> const & B)
	{
		self_.NextTimeStepHalf(E, B);
	}
	void NextTimeStepHalf_(Bool2Type<true>, Field<mesh_type, EDGE, scalar_type> const & E,
	        Field<mesh_type, FACE, scalar_type> const & B)
	{
	}
	void NextTimeStepHalf_(Bool2Type<false>, Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
	{
	}
	void NextTimeStepHalf_(Bool2Type<true>, Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
	{
		self_.NextTimeStepHalf(E, B);
	}

	particle_type self_;

	Field<mesh_type, EDGE, scalar_type> dummy_J;
	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> dummy_Jv;

};
template<typename TParticle, typename TDict, typename ...Args>
std::shared_ptr<typename ParticleWrap<TParticle>::base_type> CreateParticle(typename TParticle::mesh_type const & mesh,
        TDict const & dict, Args const & ... args)
{
	typedef typename ParticleWrap<TParticle>::base_type base_type;

	std::shared_ptr<base_type> res(nullptr);

	if (dict["Type"].template as<std::string>("Default") == TParticle::GetTypeAsString()
	        && dict["EnableImplicit"].template as<bool>(false) == TParticle::EnableImplicit)
	{
		res = std::dynamic_pointer_cast<base_type>(
		        std::shared_ptr<ParticleWrap<TParticle>>(
		                new ParticleWrap<TParticle>(mesh, dict, std::forward<Args const &>(args)...)));
	}
	return res;
}
}
// namespace simpla

#endif /* PARTICLE_BASE_H_ */
