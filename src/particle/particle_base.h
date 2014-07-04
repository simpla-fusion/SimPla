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
#include "../model/model.h"
#include "../model/surface.h"
#include "../utilities/factory.h"
namespace simpla
{

//*******************************************************************************************************
template<typename TM>
struct ParticleBase
{

public:

	typedef TM mesh_type;

	typedef typename mesh_type::scalar_type scalar_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;

	ParticleBase()
	{
	}

	virtual ~ParticleBase()
	{
	}

	//interface

	virtual std::string Save(std::string const & path, bool is_verbose) const=0;

	virtual Real GetMass() const=0;

	virtual Real GetCharge() const=0;

	virtual bool EnableImplicit() const = 0;

	virtual std::string GetTypeAsString() const = 0;

	virtual typename mesh_type:: template field<VERTEX, scalar_type> const& n() const=0;
	virtual typename mesh_type:: template field<EDGE, scalar_type> const&J() const=0;
	virtual typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const&Jv() const= 0;

	virtual void NextTimeStepZero(typename mesh_type:: template field<EDGE, scalar_type> const & E,
	        typename mesh_type:: template field<FACE, scalar_type> const & B)
	{
	}
	virtual void NextTimeStepZero(typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
	{
	}
	virtual void NextTimeStepHalf(typename mesh_type:: template field<EDGE, scalar_type> const & E,
	        typename mesh_type:: template field<FACE, scalar_type> const & B)
	{
	}
	virtual void NextTimeStepHalf(typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
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

	ParticleWrap(std::shared_ptr<particle_type> p)
			: self_(p), dummy_J(p->mesh), dummy_Jv(p->mesh)
	{
	}
	~ParticleWrap()
	{
	}

	template<typename ...Args>
	static int Register(Factory<std::string, ParticleBase<mesh_type>, Args ...>* factory)
	{
		typename Factory<std::string, ParticleBase<mesh_type>, Args ...>::create_fun_callback call_back;

		call_back = []( Args && ...args)
		{
			return std::dynamic_pointer_cast<base_type>(
					std::shared_ptr<ParticleWrap<particle_type>>(
							new ParticleWrap<particle_type>(particle_type::Create( std::forward<Args>(args)...))));
		};

		return factory->Register(particle_type::GetTypeAsString(), call_back);
	}

	Real GetMass() const
	{
		return self_->m;
	}

	Real GetCharge() const
	{
		return self_->q;
	}

	bool EnableImplicit() const
	{
		return self_->EnableImplicit;
	}
	std::string GetTypeAsString() const
	{
		return self_->GetTypeAsString();
	}
	typename mesh_type:: template field<VERTEX, scalar_type> const& n() const
	{
		return self_->n;
	}
	typename mesh_type:: template field<EDGE, scalar_type> const& J() const
	{
		return J_(std::integral_constant<bool, particle_type::EnableImplicit>());
	}
	typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const&Jv() const
	{
		return Jv_(std::integral_constant<bool, particle_type::EnableImplicit>());
	}

	void NextTimeStepZero(typename mesh_type:: template field<EDGE, scalar_type> const & E,
	        typename mesh_type:: template field<FACE, scalar_type> const & B)
	{
		NextTimeStepZero_(std::integral_constant<bool, particle_type::EnableImplicit>(), E, B);
	}
	void NextTimeStepZero(typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
	{
		NextTimeStepZero_(std::integral_constant<bool, particle_type::EnableImplicit>(), E, B);
	}
	void NextTimeStepHalf(typename mesh_type:: template field<EDGE, scalar_type> const & E,
	        typename mesh_type:: template field<FACE, scalar_type> const & B)
	{
		NextTimeStepHalf_(std::integral_constant<bool, particle_type::EnableImplicit>(), E, B);
	}
	void NextTimeStepHalf(typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
	{
		NextTimeStepHalf_(std::integral_constant<bool, particle_type::EnableImplicit>(), E, B);
	}

	std::string Save(std::string const & path, bool is_verbose = false) const
	{
		return self_->Save(path, is_verbose);
	}

private:

//	typename std::enable_if<!particle_type::EnableImplicit, typename mesh_type:: template field < EDGE, scalar_type> const&>::type J_() const
//	{
//		return self_->J;
//	}
//
//	typename std::enable_if<particle_type::EnableImplicit, typename mesh_type:: template field < EDGE, scalar_type> const&>::type J_() const
//	{
//		return dummy_J;
//	}
//
//	typename std::enable_if<particle_type::EnableImplicit, typename mesh_type:: template field < VERTEX, nTuple<3, scalar_type>> const&>::type Jv_() const
//	{
//		return self_->J;
//	}
//
//	typename std::enable_if<!particle_type::EnableImplicit, typename mesh_type:: template field < VERTEX, nTuple<3, scalar_type>> const&>::type Jv_() const
//	{
//		return dummy_Jv;
//	}

	typename mesh_type:: template field<EDGE, scalar_type> const& J_(std::integral_constant<bool, false>) const
	{
		return self_->J;
	}

	typename mesh_type:: template field<EDGE, scalar_type> const& J_(std::integral_constant<bool, true>) const
	{
		return dummy_J;
	}

	typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const& Jv_(
	        std::integral_constant<bool, true>) const
	{
		return self_->J;
	}

	typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const& Jv_(
	        std::integral_constant<bool, false>) const
	{
		return dummy_Jv;
	}
	void NextTimeStepZero_(std::integral_constant<bool, true>,
	        typename mesh_type:: template field<EDGE, scalar_type> const & E,
	        typename mesh_type:: template field<FACE, scalar_type> const & B)
	{
	}
	void NextTimeStepZero_(std::integral_constant<bool, false>,
	        typename mesh_type:: template field<EDGE, scalar_type> const & E,
	        typename mesh_type:: template field<FACE, scalar_type> const & B)
	{
		self_->NextTimeStepZero(E, B);
	}
	void NextTimeStepZero_(std::integral_constant<bool, true>,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
	{
		self_->NextTimeStepZero(E, B);
	}
	void NextTimeStepZero_(std::integral_constant<bool, false>,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
	{
	}

	void NextTimeStepHalf_(std::integral_constant<bool, false>,
	        typename mesh_type:: template field<EDGE, scalar_type> const & E,
	        typename mesh_type:: template field<FACE, scalar_type> const & B)
	{
		self_->NextTimeStepHalf(E, B);
	}
	void NextTimeStepHalf_(std::integral_constant<bool, true>,
	        typename mesh_type:: template field<EDGE, scalar_type> const & E,
	        typename mesh_type:: template field<FACE, scalar_type> const & B)
	{
	}
	void NextTimeStepHalf_(std::integral_constant<bool, false>,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
	{
	}
	void NextTimeStepHalf_(std::integral_constant<bool, true>,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
	{
		self_->NextTimeStepHalf(E, B);
	}

	std::shared_ptr<particle_type> self_;

	typename mesh_type:: template field<EDGE, scalar_type> dummy_J;
	typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> dummy_Jv;

};

}
// namespace simpla

#endif /* PARTICLE_BASE_H_ */
