/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <cstddef>
#include <exception>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../fetl/fetl.h"
# include "../fetl/field_rw_cache.h"

#include "../utilities/log.h"
#include "../utilities/memory_pool.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/type_utilites.h"
#include "../utilities/parallel.h"

#include "../io/data_stream.h"

#include "particle_base.h"
#include "particle_boundary.h"
#include "particle_pool.h"
#include "load_particle.h"
#include "save_particle.h"

#include "../modeling/command.h"

namespace simpla
{

//*******************************************************************************************************
template<class Engine, typename TStorage = ParticlePool<typename Engine::mesh_type, typename Engine::Point_s>>
class Particle: public Engine, public TStorage, public ParticleBase<typename Engine::mesh_type>
{
	std::mutex write_lock_;

public:
	static constexpr int IForm = VERTEX;

	typedef Engine engine_type;

	typedef TStorage storage_type;

	typedef ParticleBase<typename Engine::mesh_type> base_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef typename engine_type::Point_s particle_type;

	typedef typename engine_type::scalar_type scalar_type;

	typedef particle_type value_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	//container

	typedef std::list<value_type, FixedSmallSizeAlloc<value_type> > cell_type;

	typedef std::vector<cell_type> container_type;

	typedef typename container_type::iterator iterator;

	typedef typename container_type::const_iterator const_iterator;

	typedef typename cell_type::allocator_type allocator_type;

public:
	mesh_type const & mesh;
	//***************************************************************************************************
	// Constructor
	template<typename TDict, typename ...Args> Particle(mesh_type const & pmesh, TDict const & dict,
	        Args const & ...others);

	// Destructor
	virtual ~Particle();

	template<typename TDict, typename ... Others>
	void AddCommand(TDict const & dict, Material<mesh_type> const &, Others const & ...);

	static std::string GetTypeAsString()
	{
		return engine_type::GetTypeAsString();
	}
	//***************************************************************************************************
	// Interface

	std::string GetTypeAsString_() const
	{
		return GetTypeAsString();
	}
	inline Real GetMass() const
	{
		return engine_type::GetMass();
	}

	inline Real GetCharge() const
	{
		return engine_type::GetCharge();
	}

	bool EnableImplicit() const
	{
		return engine_type::EnableImplicit();
	}

	void NextTimeStepZero(Field<mesh_type, EDGE, scalar_type> const &E, Field<mesh_type, FACE, scalar_type> const & B)
	{
		if (!EnableImplicit())
		{
			NextTimeStepZero_(E, B, &J_);
		}
		else
		{
			NextTimeStepZero_(E, B, &Jv_);
		}
	}
	void NextTimeStepZero(Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
	{
		if (!EnableImplicit())
		{
			NextTimeStepZero_(E, B, &J_);
		}
		else
		{
			NextTimeStepZero_(E, B, &Jv_);
		}
	}
	void NextTimeStepHalf(Field<mesh_type, EDGE, scalar_type> const &E, Field<mesh_type, FACE, scalar_type> const & B)
	{
		NextTimeStepHalf_(E, B);
	}
	void NextTimeStepHalf(Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
	{
		NextTimeStepHalf_(E, B);
	}
	template<typename TE, typename TB, typename TJ>
	void NextTimeStepZero_(TE const & E, TB const & B, TJ *);
	template<typename TE, typename TB>
	void NextTimeStepHalf_(TE const & E, TB const & B);

	std::string Dump(std::string const & path, bool is_verbose = false) const;

	Field<mesh_type, VERTEX, scalar_type> & n()
	{
		return n_;
	}
	Field<mesh_type, VERTEX, scalar_type> const& n() const
	{
		return n_;
	}
	Field<mesh_type, EDGE, scalar_type> &J()
	{
		return J_;
	}
	Field<mesh_type, EDGE, scalar_type> const&J() const
	{
		return J_;
	}
	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> &Jv()
	{
		return Jv_;
	}
	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const&Jv() const
	{
		return Jv_;
	}

//***************************************************************************************************
	template<int IFORM, typename ...Args>
	void Scatter(Field<mesh_type, IFORM, scalar_type> *J, Args const & ... args) const;

private:

	Field<mesh_type, VERTEX, scalar_type> n_;

	Field<mesh_type, EDGE, scalar_type> J_;

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> Jv_;

	std::list<std::function<void()> > commands_;
};

template<typename Engine, typename TStorage>
template<typename TDict, typename ...Others>
Particle<Engine, TStorage>::Particle(mesh_type const & pmesh, TDict const & dict, Others const & ...others)

		: engine_type(pmesh, dict, std::forward<Others const&>(others)...),

		storage_type(pmesh, dict),

		mesh(pmesh), n_(mesh), J_(mesh), Jv_(mesh)
{
	n_.Clear();

	if (engine_type::EnableImplicit())
	{
		Jv_.Clear();
	}
	else
	{
		J_.Clear();
	}

	LoadParticle(this, dict, std::forward<Others const &>(others)...);

	AddCommand(dict["Commands"], std::forward<Others const &>(others)...);

}

template<typename Engine, typename TStorage>
Particle<Engine, TStorage>::~Particle()
{
}
template<typename Engine, typename TStorage>
template<typename TDict, typename ...Others> void Particle<Engine, TStorage>::AddCommand(TDict const & dict,
        Material<mesh_type> const & model, Others const & ...others)
{
	if (!dict.is_table())
		return;
	for (auto item : dict)
	{
		auto dof = item.second["DOF"].template as<std::string>("");

		if (dof == "n")
		{

			LOGGER << "Add constraint to " << dof;

			commands_.push_back(
			        Command<decltype(n_)>::Create(&n_, item.second, model, std::forward<Others const &>(others)...));

		}
		else if (dof == "J")
		{

			LOGGER << "Add constraint to " << dof;

			if (!J_.empty())
			{
				commands_.push_back(
				        Command<decltype(J_)>::Create(&J_, item.second, model,
				                std::forward<Others const &>(others)...));
			}
			else if (!Jv_.empty())
			{
				commands_.push_back(
				        Command<decltype(Jv_)>::Create(&Jv_, item.second, model,
				                std::forward<Others const &>(others)...));

			}
		}
		else if (dof == "ParticlesBoundary")
		{

			LOGGER << "Add constraint to " << dof;

			commands_.push_back(
			        BoundaryCondition<this_type>::Create(this, item.second, model,
			                std::forward<Others const &>(others)...));
		}
		else
		{
			UNIMPLEMENT2("Unknown DOF!");
		}
		LOGGER << DONE;
	}

}

//*************************************************************************************************

template<typename Engine, typename TStorage>
std::string Particle<Engine, TStorage>::Dump(std::string const & path, bool is_verbose) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.OpenGroup(path );

	if (is_verbose)
	{

		os

		<< engine_type::Dump(path, is_verbose)

//		<< "\n, particles = " << storage_type::Dump(*this, "particles", !is_verbose)

		        ;
	}

	os << "\n, n =" << simpla::Dump(n_, "n", is_verbose);

	if (EnableImplicit())
	{
		os << "\n, Jv =" << simpla::Dump(Jv_, "Jv", is_verbose);
	}
	else
	{
		os << "\n, J =" << simpla::Dump(J_, "J", is_verbose);
	}

	return os.str();
}

template<typename Engine, typename TStorage>
template<typename TE, typename TB, typename TJ>
void Particle<Engine, TStorage>::NextTimeStepZero_(TE const & E, TB const & B, TJ * J)
{

	if (J->empty())
		return;

	LOGGER << "Push particles to zero step [ " << engine_type::GetTypeAsString() << std::boolalpha
	        << " , Enable Implicit =" << EnableImplicit() << " ]";

	Real dt = mesh.GetDt();

	J->Clear();

	ParallelDo(

	[&](int t_num,int t_id)
	{
		for(auto s: this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			J->lock();
			for (auto & p : this->at(s) )
			{
				this->engine_type::NextTimeStepZero(&p,dt ,J,E,B);
			}
			J->unlock();
		}

	});

	storage_type::Sort();
	LOGGER << DONE;
	auto & Js = *J;
	LOG_CMD(n_ -= Diverge(MapTo<EDGE>(Js)) * dt);

}

template<typename Engine, typename TStorage>
template<typename TE, typename TB>
void Particle<Engine, TStorage>::NextTimeStepHalf_(TE const & E, TB const & B)
{

	LOGGER << "Push particles to half step[ " << engine_type::GetTypeAsString() << std::boolalpha
	        << " , Enable Implicit =" << EnableImplicit() << " ]";

	Real dt = mesh.GetDt();

	ParallelDo(

	[&](int t_num,int t_id)
	{
		for(auto s: this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			for (auto & p : this->at(s) )
			{
				this->engine_type::NextTimeStepHalf(&p,dt ,E,B);
			}
		}

	});

	storage_type::Sort();
	LOGGER << DONE;
}

template<typename Engine, typename TStorage> template<int IFORM, typename ...Args>
void Particle<Engine, TStorage>::Scatter(Field<mesh_type, IFORM, scalar_type> *pJ, Args const &... args) const
{
	ParallelDo(

	[&](int t_num,int t_id)
	{
		for(auto s: this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			pJ->lock();
			for (auto const& p : this->at(s) )
			{
				this->engine_type::Scatter(p,pJ,std::forward<Args const &>(args)...);
			}
			pJ->unlock();
		}

	});
}
//*************************************************************************************************
template<typename TX, typename TV, typename TE, typename TB> inline
void BorisMethod(Real dt, Real cmr, TE const & E, TB const &B, TX *x, TV *v)
{
// @ref  Birdsall(1991)   p.62
// Bories Method

	Vec3 v_;

	auto t = B * (cmr * dt * 0.5);

	(*v) += E * (cmr * dt * 0.5);

	v_ = (*v) + Cross((*v), t);

	(*v) += Cross(v_, t) * (2.0 / (Dot(t, t) + 1.0));

	(*v) += E * (cmr * dt * 0.5);

	(*x) += (*v) * dt;

}

}
// namespace simpla

#endif /* PARTICLE_H_ */
