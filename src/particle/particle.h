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
class Particle: public Engine
{

public:
	static constexpr int IForm = VOLUME;

	typedef Engine engine_type;

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
	mesh_type const &mesh;

	template<typename ...Args> Particle(mesh_type const & pmesh);

	virtual ~Particle();

	template<typename TWrap, typename TDict, typename ...Args> static //
	bool CreateWrap(TWrap* res, TDict const & dict, mesh_type const & mesh, Args const & ...args)
	{
		bool isDone = false;

		if (dict["Type"].template as<std::string>() == engine_type::TypeName())
		{

			typedef typename TWrap::TE TE;
			typedef typename TWrap::TB TB;
			typedef typename TWrap::TN TN;
			typedef typename TWrap::TJ TJ;

			auto particle = std::shared_ptr<this_type>(new this_type(mesh));

			particle->Load(dict, std::forward<Args const &>(args)...);

			using namespace std::placeholders;

			res->NextTimeStep_ = std::bind(&this_type::template NextTimeStep<TN, TJ, TE, TB>, particle, _1, _2, _3, _4,
			        _5);

			res->Print = std::bind(&this_type::Print, particle, _1);

			res->Dump = std::bind(&this_type::Dump, particle, _1, false);

			isDone = true;
		}

		return isDone;
	}

	allocator_type GetAllocator()
	{
		return pool_.get_allocator();
	}

//***************************************************************************************************

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

	template<typename ...Args> void Load(Args const &... args);

	std::ostream & Print(std::ostream & os) const;

	std::string Dump(std::string const &, bool compact_storage = false) const;

	void Update();

//***************************************************************************************************
	template<typename TN, typename TJ, typename ...Args>
	void NextTimeStep(Real dt, TN * n, TJ * J, Args const& ... args);

	void Sort();

	bool IsSorted() const
	{
		return isSorted_;
	}

	std::string GetEngineTypeAsString() const
	{
		return engine_type::GetTypeAsString();
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

	cell_type pool_;

	container_type data_;

	bool isSorted_;
	bool particleSortingIsEnable_;

	/**
	 *  resort particles in cell 's', and move out boundary particles to 'dest' container
	 * @param
	 */
	void Resort(index_type s, container_type * dest = nullptr);

	std::vector<container_type> mt_data_; // for sort

};

template<class Engine>
template<typename ...Args> Particle<Engine>::Particle(mesh_type const & pmesh)
		: engine_type(pmesh), mesh(pmesh), isSorted_(true), particleSortingIsEnable_(true)
{
}

template<class Engine>
Particle<Engine>::~Particle()
{
}

template<class Engine>
template<typename ...Args>
void Particle<Engine>::Load(Args const & ... args)
{
	Update();
	LoadParticle(this, std::forward<Args const &>(args)...);
}

template<class Engine>
std::ostream & Particle<Engine>::Print(std::ostream & os) const
{
	engine_type::Print(os);

	return os;
}
template<class Engine>
std::string Particle<Engine>::Dump(std::string const & name, bool compact_storage) const
{
	return simpla::Dump(*this, name, compact_storage);
}
template<typename TM>
std::ostream & operator<<(std::ostream & os, std::pair<std::string, Particle<TM>> const &self)
{
	return self.Save(os);
}

template<class Engine>
void Particle<Engine>::Update()
{
	if (data_.size() < mesh.GetNumOfElements(IForm))
		data_.resize(mesh.GetNumOfElements(IForm), cell_type(GetAllocator()));

	if (particleSortingIsEnable_)
	{

		const unsigned int num_threads = std::thread::hardware_concurrency();

		mt_data_.resize(num_threads);

		for (auto & d : mt_data_)
		{
			d.resize(mesh.GetNumOfElements(IForm), cell_type(GetAllocator()));
		}
	}
}

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

	Update();

	if (IsSorted())
		return;

	if (particleSortingIsEnable_)
	{

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
	}
	isSorted_ = true;
}

template<class Engine>
template<typename TN, typename TJ, typename ...Args>
void Particle<Engine>::NextTimeStep(Real dt, TN * n, TJ * J, Args const& ... args)
{

	if (data_.empty())
	{
		WARNING << "Particle [ " << engine_type::GetTypeAsString() << "] is not initialized!";
		return;
	}

	LOGGER << "Push particles [ " << engine_type::GetTypeAsString() << "]";

	Sort();

	ParallelDo(

	[& ](int t_num,int t_id)
	{
		for(auto s: this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			n->lock();
			J->lock();
			for (auto & p : this->data_.at(this->mesh.Hash(s)) )
			{
				this->engine_type::NextTimeStep(&p,dt,n,J,std::forward<Args const &>( args) ...);

			}
			J->unlock();
			n->unlock();
		}

	});

	isSorted_ = false;
	Sort();

	LOGGER << DONE;
	VERBOSE << "Particle Sorting is " << (particleSortingIsEnable_ ? "enabled" : "disabled") << ".";

}

template<typename TF, typename TX, typename TV>
void ScatterTo(TX const & x, TV const &v, TF *f)
{
	f->mesh.Scatter(x, v, f);
}

//******************************************************************************************************
template<typename TX, typename TV, typename FE, typename FB> inline
void BorisMethod(Real dt, Real cmr, FE const & fE, FB const &fB, TX *x, TV *v)
{
// @ref  Birdsall(1991)   p.62
// Bories Method

	(*x) += (*v) * 0.5 * dt;

	Vec3 v_;
	auto B = real(fB((*x)));
	auto E = real(fE((*x)));
	auto t = B * (cmr * dt * 0.5);

	(*v) += E * (cmr * dt * 0.5);

	v_ = (*v) + Cross((*v), t);

	(*v) += Cross(v_, t) * (2.0 / (Dot(t, t) + 1.0));

	(*v) += E * (cmr * dt * 0.5);

	(*x) += (*v) * 0.5 * dt;

}

}
// namespace simpla

#endif /* PARTICLE_H_ */
