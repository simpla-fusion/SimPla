/*
 * particle_base.h
 *
 *  Created on: 2014年1月16日
 *      Author: salmon
 */

#ifndef PARTICLE_BASE_H_
#define PARTICLE_BASE_H_

#include <iostream>
#include <memory>
#include <string>

#include "../fetl/field.h"
#include "../fetl/save_field.h"
#include "../fetl/primitives.h"
#include "../io/data_stream.h"

namespace simpla
{
template<typename Engine> class Particle;
template<typename TE>
std::ostream & operator<<(std::ostream & os, Particle<TE> const &self)
{
	return self.Print(os);
}
//*******************************************************************************************************
template<typename TM>
struct ParticleBase
{

public:
	typedef TM mesh_type;

	typedef typename mesh_type::scalar_type scalar_type;

	mesh_type const &mesh;

	Field<mesh_type, VERTEX, scalar_type> n;
	Field<mesh_type, EDGE, scalar_type> J;

	ParticleBase(mesh_type const &pmesh) :
			mesh(pmesh), n(mesh), J(mesh)
	{
		n.Clear();
		J.Clear();
	}
	virtual ~ParticleBase()
	{
	}

	virtual Real GetMass() const=0;

	virtual Real GetCharge() const=0;

	virtual void NextTimeStep(Real dt,
			Field<mesh_type, EDGE, scalar_type> const & E,
			Field<mesh_type, FACE, scalar_type> const & B)=0;

	virtual void Print(std::ostream & os) const=0;

	virtual void Dump(std::string const & name, bool compact_storage) const
	{
		LOGGER << simpla::Dump(n, name + "_n", compact_storage);
		LOGGER << simpla::Dump(J, name + "_J", compact_storage);
	}

};

//template<typename TP>
//struct ParticleWrap: public ParticleBase<typename TP::mesh_type>
//{
//public:
//
//	typedef TP particle_type;
//
//	typedef typename TP::mesh_type mesh_type;
//
//	typedef typename mesh_type::scalar_type scalar_type;
//
//	typedef ParticleBase<mesh_type> base_type;
//
//	typedef ParticleWrap<particle_type> this_type;
//private:
//	std::shared_ptr<particle_type> p_;
//public:
//	template<typename ...Args>
//	ParticleWrap(mesh_type const & mesh, Args const & ... args)
//			: base_type(mesh), p_(new particle_type(mesh, std::forward<Args const &>(args)...))
//	{
//	}
//	~ParticleWrap()
//	{
//	}
//
//	template<typename ...Args>
//	static std::shared_ptr<base_type> Create(std::string const & type_str, Args const & ... args)
//	{
//		std::shared_ptr<base_type> res(nullptr);
//
//		if (type_str == particle_type::GetTypeAsString())
//			res = std::dynamic_pointer_cast<base_type>(
//			        std::shared_ptr<this_type>(new this_type(std::forward<Args const &>(args)...)));
//
//		return res;
//	}
//
//	inline Real GetMass() const
//	{
//		return p_->GetMass();
//	}
//
//	inline Real GetCharge() const
//	{
//		return p_->GetCharge();
//	}
//
//	void NextTimeStep(Real dt, Field<mesh_type, EDGE, scalar_type> const E,
//	        Field<mesh_type, FACE, scalar_type> const & B)
//	{
//		p_->NextTimeStep(dt, &(base_type::n), &(base_type::J), E, B);
//	}
//
//	void Print(std::ostream & os) const
//	{
//		p_->Print(os);
//	}
//
//	std::string Dump(std::string const & name, bool compact_storage) const
//	{
//		return p_->Dump(name, compact_storage);
//	}
//
//};
//*******************************************************************************************************

}// namespace simpla

#endif /* PARTICLE_BASE_H_ */
