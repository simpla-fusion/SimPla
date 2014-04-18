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
#include "load_particle.h"
#include "save_particle.h"

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

	typedef typename cell_type::iterator iterator;

	typedef typename cell_type::allocator_type allocator_type;

	typedef std::vector<cell_type> container_type;

public:
	mesh_type const & mesh;
	//***************************************************************************************************
	// Constructor
	template<typename ...Args> Particle(mesh_type const & pmesh, Args const & ...args);

	// Destructor
	virtual ~Particle();

	//***************************************************************************************************
	// Interface

	static std::string GetTypeAsString()
	{
		return engine_type::GetTypeAsString();
	}
	inline Real GetMass() const
	{
		return engine_type::GetMass();
	}

	inline Real GetCharge() const
	{
		return engine_type::GetCharge();
	}
	void NextTimeStep(Real dt, Field<mesh_type, EDGE, scalar_type> const &E,
	        Field<mesh_type, FACE, scalar_type> const & B);

	std::string Dump(std::string const & path, bool is_verbose = false) const;

	template<int IFORM, typename ...Args>
	void Scatter(Field<mesh_type, IFORM, scalar_type> *J, Args const & ... args) const;

	//***************************************************************************************************

	allocator_type GetAllocator()
	{
		return pool_.get_allocator();
	}

	inline void Insert(index_type s, typename engine_type::Point_s && p)
	{
		data_[mesh.Hash(s)].emplace_back(p);
	}

	cell_type & operator[](size_t s)
	{
		return data_.at(s);
	}
	cell_type const & operator[](size_t s) const
	{
		return data_.at(s);
	}

//***************************************************************************************************

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
		particleSortingIsEnable_ = f;
	}
	bool GetParticleSorting() const
	{
		return particleSortingIsEnable_;
	}

	container_type const & GetTree() const
	{
		return data_;
	}
private:

	bool isSorted_;
	bool particleSortingIsEnable_;
	cell_type pool_;
	container_type data_;

	/**
	 *  resort particles in cell 's', and move out boundary particles to 'dest' container
	 * @param
	 */
	void Resort(index_type s, container_type * dest = nullptr);

	std::vector<container_type> mt_data_; // for sort

};

template<class Engine>
template<typename ...Args>
Particle<Engine>::Particle(mesh_type const & pmesh, Args const & ...args)
		: engine_type(pmesh, std::forward<Args const&>(args)...),

		base_type(pmesh),

		mesh(pmesh),

		isSorted_(true), particleSortingIsEnable_(true),

		pool_(),

		data_(pmesh.GetNumOfElements(IForm), cell_type(GetAllocator()))
{

	if (particleSortingIsEnable_)
	{

		const unsigned int num_threads = std::thread::hardware_concurrency();

		mt_data_.resize(num_threads);

		for (auto & d : mt_data_)
		{
			d.resize(mesh.GetNumOfElements(IForm), cell_type(GetAllocator()));
		}
	}

	LoadParticle(this, std::forward<Args const &>(args)...);
}

template<class Engine>
Particle<Engine>::~Particle()
{
}

//*************************************************************************************************

template<class Engine>
std::string Particle<Engine>::Dump(std::string const & path, bool is_verbose) const
{
	std::stringstream os;

	if (is_verbose)
	{
		GLOBAL_DATA_STREAM.OpenGroup(path );

		os

		<< engine_type::Dump(path,is_verbose)

		<< "\n, particles = " << simpla::Dump(*this, "particles", !is_verbose);
	}

	os << base_type::Dump(path, is_verbose);

	return os.str();
}

#define DISABLE_MULTI_THREAD

template<class Engine>
void Particle<Engine>::NextTimeStep(Real dt, Field<mesh_type, EDGE, scalar_type> const & E,
        Field<mesh_type, FACE, scalar_type> const & B)
{
	if (data_.empty())
	{
		WARNING << "Particle [ " << engine_type::GetTypeAsString() << "] is not initialized!";
		return;
	}

	LOGGER << "Push particles [ " << engine_type::GetTypeAsString() << "]";

	Sort();

	base_type::J.Clear();

	ParallelDo(

	[&](int t_num,int t_id)
	{
		for(auto s: this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			this->J.lock();
			for (auto & p : this->data_.at(this->mesh.Hash(s)) )
			{
				this->engine_type::NextTimeStep(&p,dt ,&(this->base_type::J),E,B);

			}
			this->J.unlock();
		}

	});

	base_type::n -= Diverge(base_type::J) * dt;

	isSorted_ = false;
	Sort();

	LOGGER << DONE;
}
template<class Engine> template<int IFORM, typename ...Args>
void Particle<Engine>::Scatter(Field<mesh_type, IFORM, scalar_type> *pJ, Args const &... args) const
{
	ParallelDo(

	[&](int t_num,int t_id)
	{
		for(auto s: this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			pJ->lock();
			for (auto const& p : this->data_.at(this->mesh.Hash(s)) )
			{
				this->engine_type::Scatter(p,pJ,std::forward<Args const &>(args)...);
			}
			pJ->unlock();
		}

	});
}
//*************************************************************************************************
template<class Engine>
void Particle<Engine>::Resort(index_type id_src, container_type *other)
{
	try
	{

		auto & cell = this->data_.at(this->mesh.Hash(id_src));

		auto pt = cell.begin();

		while (pt != cell.end())
		{
			auto p = pt;
			++pt;

			index_type id_dest = mesh.CoordinatesGlobalToLocal(&(p->x));

			p->x = mesh.CoordinatesLocalToGlobal(id_dest, p->x);

			if (!(id_dest == id_src))
			{

				(*other).at(this->mesh.Hash(id_dest)).splice((*other).at(this->mesh.Hash(id_dest)).begin(), cell, p);

			}

		}
	} catch (std::out_of_range const & e)
	{
		ERROR << "out of range!";
	}
}

template<class Engine>
void Particle<Engine>::Sort()
{

	if (IsSorted() || !particleSortingIsEnable_)
		return;

	VERBOSE << "Particle sorting is enabled!";

	ParallelDo(

	[this](int t_num,int t_id)
	{
		for(auto s:this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			this->Resort(s, &(this->mt_data_[t_id]));
		}
	}

	);

	ParallelDo(

	[this](int t_num,int t_id)
	{
		for(auto s:this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			auto idx = this->mesh.Hash(s);

			this->data_.at(idx) .splice(this->data_.at(idx).begin(), this->mt_data_[t_id].at(idx));
		}
	}

	);

	isSorted_ = true;
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
