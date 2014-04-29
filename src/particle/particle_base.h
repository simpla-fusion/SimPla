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

	std::string GetTypeAsString() const
	{
		return GetTypeAsString_();
	}

	//interface
	virtual Field<mesh_type, VERTEX, scalar_type> & n()=0;
	virtual Field<mesh_type, VERTEX, scalar_type> const& n() const=0;
	virtual Field<mesh_type, EDGE, scalar_type> &J()=0;
	virtual Field<mesh_type, EDGE, scalar_type> const&J() const=0;
	virtual Field<mesh_type, VERTEX, nTuple<3, scalar_type>> &Jv() = 0;
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

	virtual std::string Dump(std::string const & path, bool is_verbose) const=0;

	virtual std::string GetTypeAsString_() const=0;

	virtual Real GetMass() const=0;

	virtual Real GetCharge() const=0;

	virtual bool EnableImplicit() const
	{
		return false;
	}

	virtual void Accept(VisitorBase const& visitor)
	{
		visitor.Visit(this);
	}
	virtual void Add(index_type s, std::function<Real(coordinates_type *, nTuple<3, Real>*)> const & generator)
	{
	}
	virtual void Clear(index_type s)
	{
	}
	virtual void Remove(index_type s,
	        std::function<bool(coordinates_type const&, nTuple<3, Real> const&)> const & filter)
	{
	}

	virtual void Modify(index_type s, std::function<void(coordinates_type *, nTuple<3, Real>*)> const & op)
	{
	}

	virtual void Traversal(index_type s,
	        std::function<void(scalar_type, coordinates_type const&, nTuple<3, Real> const&)> const & op)
	{
	}
};

}
// namespace simpla

#endif /* PARTICLE_BASE_H_ */
