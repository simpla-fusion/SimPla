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
#include "load_particle.h"
#include "save_particle.h"

#include "../modeling/command.h"

#ifndef NO_STD_CXX
//need  libstdc++
#include <ext/mt_allocator.h>
template<typename T> using FixedSmallSizeAlloc=__gnu_cxx::__mt_alloc<T>;
#endif

namespace simpla
{

//*******************************************************************************************************
template<class Engine>
class Particle: public Engine, public ParticleBase<typename Engine::mesh_type>
{
	std::mutex write_lock_;

public:
	static constexpr int IForm = VERTEX;

	typedef Engine engine_type;

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
	template<typename TDict, typename ...Args> Particle(mesh_type const & pmesh,
			TDict const & dict, Args const & ...others);

	// Destructor
	virtual ~Particle();

	template<typename TDict, typename ... Others>
	void AddCommand(TDict const & dict, Material<mesh_type> const &,
			Others const & ...);

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

	void NextTimeStep(Field<mesh_type, EDGE, scalar_type> const &E,
			Field<mesh_type, FACE, scalar_type> const & B);

	template<typename TJ>
	void NextTimeStep(TJ * J, Field<mesh_type, EDGE, scalar_type> const & E,
			Field<mesh_type, FACE, scalar_type> const & B);

	std::string Dump(std::string const & path, bool is_verbose = false) const;

	void Clear(index_type s);

	void Add(index_type s, cell_type &&);

	void Add(index_type s,
			std::function<Real(coordinates_type *, nTuple<3, Real>*)> const & generator);

	void Remove(index_type s,
			std::function<bool(coordinates_type const&, nTuple<3, Real> const&)> const & filter);

	void Modify(index_type s,
			std::function<void(coordinates_type *, nTuple<3, Real>*)> const & foo);

	void Traversal(index_type s,
			std::function<
					void(scalar_type, coordinates_type const&,
							nTuple<3, Real> const&)> const & op);

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
	inline void Insert(index_type s, typename engine_type::Point_s p)
	{
		this->at(s).emplace_back(p);
	}
	allocator_type GetAllocator()
	{
		return allocator_;
	}

	cell_type & operator[](index_type s)
	{
		return data_[mesh.Hash(s)];
	}
	cell_type const & operator[](index_type s) const
	{
		return data_[mesh.Hash(s)];
	}
	cell_type &at(index_type s)
	{
		return data_.at(mesh.Hash(s));
	}
	cell_type const & at(index_type s) const
	{
		return data_.at(mesh.Hash(s));
	}

//	iterator begin()
//	{
//		return data_.begin();
//	}
//
//	iterator end()
//	{
//		return data_.end();
//	}
//
//	const_iterator begin() const
//	{
//		return data_.begin();
//	}
//
//	const_iterator end() const
//	{
//		return data_.end();
//	}
//***************************************************************************************************
	template<int IFORM, typename ...Args>
	void Scatter(Field<mesh_type, IFORM, scalar_type> *J,
			Args const & ... args) const;

	void Sort();

	bool IsSorted() const
	{
		return isSorted_;
	}

	size_t size() const
	{
		size_t res = 0;

		for (auto const & v : data_)
		{
			res += v.size();
		}
		return res;
	}
	void SetParticleSorting(bool f)
	{
		enableSorting_ = f;
	}
	bool GetParticleSorting() const
	{
		return enableSorting_;
	}

	container_type const & GetTree() const
	{
		return data_;
	}

private:

	bool isSorted_;
	bool enableSorting_;

	allocator_type allocator_;

	container_type data_;

	Field<mesh_type, VERTEX, scalar_type> n_;

	Field<mesh_type, EDGE, scalar_type> J_;

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> Jv_;

	/**
	 *  resort particles in cell 's', and move out boundary particles to 'dest' container
	 * @param
	 */
	template<typename TDest> void Sort(index_type id_src, TDest *dest);

	std::list<std::function<void()> > commands_;
};

template<class Engine>
template<typename TDict, typename ...Others>
Particle<Engine>::Particle(mesh_type const & pmesh, TDict const & dict,
		Others const & ...others)

:
		engine_type(pmesh, dict, std::forward<Others const&>(others)...),

		mesh(pmesh), isSorted_(false),

		data_(mesh.GetNumOfElements(IForm), cell_type(GetAllocator()))

		, n_(mesh), J_(mesh), Jv_(mesh)
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

	enableSorting_ = dict["EnableSorting"].template as<bool>(false);

	AddCommand(dict["Commands"], std::forward<Others const &>(others)...);

}

template<class Engine>
Particle<Engine>::~Particle()
{
}
template<class Engine>
template<typename TDict, typename ...Others> void Particle<Engine>::AddCommand(
		TDict const & dict, Material<mesh_type> const & model,
		Others const & ...others)
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
					Command<decltype(n_)>::Create(&n_, item.second, model,
							std::forward<Others const &>(others)...));

		}
		else if (dof == "J")
		{

			LOGGER << "Add constraint to " << dof;

			commands_.push_back(
					Command<decltype(J_)>::Create(&J_, item.second, model,
							std::forward<Others const &>(others)...));

		}
		else if (dof == " Jv")
		{

			LOGGER << "Add constraint to " << dof;

			commands_.push_back(
					Command<decltype(Jv_)>::Create(&Jv_, item.second, model,
							std::forward<Others const &>(others)...));

		}
//		else if (dof == "Particles")
//		{
//			commands_.push_back(
//					Command<this_type>::Create(this, item.second, model_));
//		}
		else if (dof == "ParticlesBoundary")
		{

			LOGGER << "Add constraint to " << dof;

			commands_.push_back(
					BoundaryCondition<this_type>::Create(this, item.second,
							model, std::forward<Others const &>(others)...));
		}
		else
		{
			UNIMPLEMENT2("Unknown DOF!");
		}
		LOGGER << DONE;
	}

}

//*************************************************************************************************

template<class Engine>
std::string Particle<Engine>::Dump(std::string const & path,
		bool is_verbose) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.OpenGroup(path );

	if (is_verbose)
	{

		os

		<< engine_type::Dump(path, is_verbose)

		<< "\n, particles = " << simpla::Dump(*this, "particles", !is_verbose);
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

#define DISABLE_MULTI_THREAD

template<class Engine>

void Particle<Engine>::NextTimeStep(
		Field<mesh_type, EDGE, scalar_type> const & E,
		Field<mesh_type, FACE, scalar_type> const & B)
{
	if (EnableImplicit())
	{
		NextTimeStep(&Jv_, E, B);
	}
	else
	{
		NextTimeStep(&J_, E, B);
	}
}

template<class Engine>
template<typename TJ>
void Particle<Engine>::NextTimeStep(TJ * J,
		Field<mesh_type, EDGE, scalar_type> const & E,
		Field<mesh_type, FACE, scalar_type> const & B)
{
	if (data_.empty())
	{
		WARNING << "Particle [ " << engine_type::GetTypeAsString()
				<< "] is not initialized!";
		return;
	}

	LOGGER << "Push particles [ " << engine_type::GetTypeAsString()
			<< std::boolalpha << " , Enable Implicit =" << EnableImplicit()
			<< " , Enable Sorting =" << enableSorting_ << " ]";

	Real dt = mesh.GetDt();

	Sort();

	J->Clear();

	ParallelDo(

	[&](int t_num,int t_id)
	{
		for(auto s: this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			J->lock();
			for (auto & p : this->at(s) )
			{
				this->engine_type::NextTimeStep(&p,dt ,J,E,B);
			}
			J->unlock();
		}

	});

	n_ -= Diverge(MapTo<EDGE>(*J)) * dt;

	isSorted_ = false;
	Sort();

	for (auto const & comm : commands_)
	{
		comm();
	}

	LOGGER << DONE;
}

template<class Engine> template<int IFORM, typename ...Args>
void Particle<Engine>::Scatter(Field<mesh_type, IFORM, scalar_type> *pJ,
		Args const &... args) const
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
template<class Engine>
template<typename TDest>
void Particle<Engine>::Sort(index_type id_src, TDest *dest)
{

	auto & src = this->at(id_src);

	auto pt = src.begin();

	while (pt != src.end())
	{
		auto p = pt;
		++pt;

		index_type id_dest = mesh.CoordinatesGlobalToLocalDual(&(p->x));

		p->x = mesh.CoordinatesLocalToGlobal(id_dest, p->x);

		if (id_dest != id_src)
		{
			(*dest)[id_dest].splice((*dest)[id_dest].begin(), src, p);
		}

	}

}

template<class Engine>
void Particle<Engine>::Sort()
{

	if (IsSorted() || !enableSorting_)
		return;

	VERBOSE << "Particle sorting is enabled!";

	ParallelDo(

	[this](int t_num,int t_id)
	{
		std::map<index_type,cell_type> dest;
		for(auto s:this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			this->Sort(s, &dest);
		}

		write_lock_.lock();
		for(auto & v :dest)
		{
			auto & c = this->at(v.first);

			c.splice(c.begin(), v.second);
		}
		write_lock_.unlock();
	}

	);

	isSorted_ = true;
}

template<class Engine>
void Particle<Engine>::Clear(index_type s)
{
	this->at(s).clear();
}
template<class Engine>
void Particle<Engine>::Add(index_type s, cell_type && other)
{
	this->at(s).slice(this->at(s).begin(), other);
}

template<class Engine>
void Particle<Engine>::Add(index_type s,
		std::function<Real(coordinates_type *, nTuple<3, Real>*)> const & gen)
{
	coordinates_type x;
	nTuple<3, Real> v;
	Real f = gen(&x, &v);
	this->at(s).push_back(engine_type::make_point(x, v, f));

}

template<class Engine>
void Particle<Engine>::Remove(index_type s,
		std::function<bool(coordinates_type const&, nTuple<3, Real> const&)> const & filter)
{
	auto & cell = this->at(s);

	auto pt = cell.begin();

	while (pt != cell.end())
	{
		coordinates_type x;
		nTuple<3, Real> v;

		engine_type::PullBack(*pt, &x, &v);
		if (filter(x, v))
		{
			pt = cell.erase(pt);
		}
		else
		{
			++pt;
		}
	}

}

template<class Engine>
void Particle<Engine>::Modify(index_type s,
		std::function<void(coordinates_type *, nTuple<3, Real>*)> const & op)
{

	for (auto & p : this->at(s))
	{
		coordinates_type x;
		nTuple<3, Real> v;

		engine_type::PullBack(p, &x, &v);
		op(&x, &v);
		engine_type::PushForward(x, v, &p);
	}
}

template<class Engine>
void Particle<Engine>::Traversal(index_type s,
		std::function<
				void(scalar_type, coordinates_type const&,
						nTuple<3, Real> const&)> const & op)
{

	for (auto const & p : this->at(s))
	{
		coordinates_type x;
		nTuple<3, Real> v;
		scalar_type f = engine_type::PullBack(p, &x, &v);
		op(f, x, v);
	}

}

//******************************************************************************************************
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
