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
#include <thread>
#include <utility>
#include <vector>

#include "../fetl/fetl.h"
# include "../fetl/field_rw_cache.h"

#include "../utilities/log.h"
#include "../utilities/lua_state.h"
#include "../utilities/memory_pool.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/type_utilites.h"
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

	enum
	{
		IForm = 3
	};

	enum
	{
		REFELECT, ABSORB
	};

	typedef Engine engine_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef typename engine_type::Point_s particle_type;

	typedef particle_type value_type;

	DEFINE_FIELDS(mesh_type)

	//container

	typedef std::list<value_type, FixedSmallSizeAlloc<value_type> > cell_type;

	typedef typename cell_type::iterator iterator;

	typedef typename cell_type::allocator_type allocator_type;

	typedef std::vector<cell_type> container_type;

public:
	mesh_type const &mesh;

	template<typename ...Args> Particle(mesh_type const & pmesh);

	virtual ~Particle();

	/**
	 *  Dump particles to a continue memory block
	 *  !!!this is a heavy operation!!!
	 *
	 * @return <datapoint , number of particles>
	 */
	std::pair<std::shared_ptr<value_type>, size_t> GetDataSet() const;

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

	void Initiallize();

	void DumpData(std::string const &path);

	void Load(LuaObject const &cfg);

	std::ostream & Save(std::ostream & os) const;

	void Update()
	{
//		Engine::Update();
	}

	//***************************************************************************************************

	template<typename ... Args> void NextTimeStep(Real dt, Args const& ... args);

	template<typename TJ, typename ... Args> void Scatter(TJ * J, Args const & ... args) const;

	void Sort();

	void Boundary()
	{
		UNIMPLEMENT;
	}
	template<typename TMaterialTag, typename ... Args>
	void Boundary(int flag, TMaterialTag in, TMaterialTag out, Real dt, Args const &... args);

	template<typename ... Args> void Collide(Args const& ... args);

	/**
	 *  resort particles in cell 's', and move out boundary particles to 'dest' container
	 * @param
	 */
	void Resort(index_type s, container_type * dest = nullptr);

	bool IsSorted() const
	{
		return isSorted_;
	}

	std::string const &GetName() const
	{
		return name_;
	}
	void SetName(std::string const & s)
	{
		name_ = s;
	}
	std::string const &name() const
	{
		return name_;
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

private:

	cell_type pool_;

	container_type data_;

	std::vector<container_type> mt_data_; // for sort

	bool isSorted_;

	std::string name_;

};

template<class Engine>
template<typename ...Args> Particle<Engine>::Particle(mesh_type const & pmesh)
		: engine_type(pmesh), mesh(pmesh), isSorted_(true), name_("unnamed")
{
}

template<class Engine>
Particle<Engine>::~Particle()
{
}

template<class Engine>
void Particle<Engine>::Load(LuaObject const &cfg)
{
	Initiallize();

	LoadParticle(cfg, this);

}
template<class Engine>
std::pair<std::shared_ptr<typename Engine::Point_s>, size_t> Particle<Engine>::GetDataSet() const
{
	size_t num = size();

	std::shared_ptr<value_type> res = (MEMPOOL.allocate_shared_ptr<value_type>(num));

	value_type * it = res.get();

	for (auto const & l : data_)
	{
		it = std::copy(l.begin(), l.end(), it);
	}

	return std::make_pair(res, num);

}

template<class Engine>
std::ostream & Particle<Engine>::Save(std::ostream & os) const
{
	os << "{ Name = '" << GetName() << "' , ";

	engine_type::Save(os);

	os << ",\n"

	<< "\tData = " << Dump(*this, this->GetName())

	<< "} ";

	return os;
}

template<typename TM>
std::ostream & operator<<(std::ostream & os, Particle<TM> const &self)
{
	return self.Serialize(os);
}

template<class Engine>
void Particle<Engine>::Initiallize()
{
	if (data_.size() < mesh.GetNumOfElements(IForm))
		data_.resize(mesh.GetNumOfElements(IForm), cell_type(GetAllocator()));

	const unsigned int num_threads = std::thread::hardware_concurrency();

	mt_data_.resize(num_threads);

	for (auto & d : mt_data_)
	{
		d.resize(mesh.GetNumOfElements(IForm), cell_type(GetAllocator()));
	}
}

template<class Engine>
void Particle<Engine>::DumpData(std::string const &path)
{
	GLOBAL_DATA_STREAM.OpenGroup(path);

	constexpr int trace_cache_depth_=10;

//	if (trace_cache_.empty())
//	{
//		size_t num_particle = trace_particle_.size();
//		trace_cache_.resize(num_particle * trace_cache_depth_);
//		trace_tail_ = trace_cache_.begin();
//	}
//
//	for (auto const& p : trace_particle_)
//	{
//		*trace_tail_ = *p;
//		++trace_tail_;
//	}
//
//	if (trace_tail_ == trace_cache_.end())
//	{
//
//		size_t dims[2] =
//		{	trace_particle_.size(), trace_cache_.size() / trace_particle_.size()};
//
//		LOGGER << Dump(&trace_cache_[0], GetName(), 2, dims, true);
//
//		trace_tail_ = trace_cache_.begin();
//	}
}

template<class Engine>
void Particle<Engine>::Resort(index_type src, container_type *other)
{
	if (other == nullptr)
		other = &(this->data_);
	auto & cell = this->data_[src];
	auto pt = cell.begin();
	while (pt != cell.end())
	{
		auto p = pt;
		++pt;

		auto dest = this->mesh.SearchCell(src, &(p->x[0]));

		if (dest != src)
		{
			(*other)[dest].splice((*other)[dest].begin(), cell, p);
		}

	}
}

template<class Engine>
void Particle<Engine>::Sort()
{

	Initiallize();

//	if (IsSorted())
//		return;
//	try
//	{
//		mesh.template Trversal<IForm>(
//
//		[&](index_type const &s, container_type * d)
//		{
//
//			Resort(s,d);
//		}, &(this->mt_data_[0]))
//
//		);
//
//		mesh.template Trversal<IForm>(
//
//		[&](index_type ss )
//		{
//
//			auto s=mesh.Hash(ss);
//
//			this->data_[s].splice(this->data_[s].begin(),this->mt_data_[i][s] );
//
//		}
//
//		);
//
//	} catch (std::exception const & e)
//	{
//		ERROR << e.what();
//	}
//
//	isSorted_ = true;
}

template<class Engine>
template<typename ...Args>
void Particle<Engine>::NextTimeStep(Real dt, Args const& ... args)
{
	if (data_.empty())
	{
		WARNING << "Particle [" << GetName() << " : " << engine_type::GetTypeAsString() << "] is not initialized!";
		return;
	}

	Sort();

	NextTimeStep(dt, std::forward<Args const&>(args) ...);

	mesh.template Traversal<IForm>(

	[&](index_type const &s, Args const & ... args2)
	{
		for (auto & p : this->data_[mesh.Hash(s)])
		{
			engine_type::NextTimeStep(&p, dt, args2...);
		}
	}, std::forward<Args const &>(args)...

	);

	isSorted_ = false;

	Sort();
	Boundary();
}

template<class Engine>
template<typename TJ, typename ...Args>
void Particle<Engine>::Scatter(TJ * J, Args const & ... args) const
{
	if (data_.empty())
	{
		WARNING << "Particle [" << GetName() << " : " << engine_type::GetTypeAsString() << "] is not initialized!";
		return;
	}

	if (!IsSorted())
		ERROR << "Particles are not sorted!";

	mesh.template Traversal<IForm>(

	[&](index_type s,TJ * J2,Args const & ... args2)
	{

		for (auto const& p : this->data_[mesh.Hash(s)])
		{
			engine_type::Scatter(p, J2 , args2...);
		}

	}, J, std::forward<Args const &>(args) ...

	);

}

template<class Engine>
template<typename TMaterialTag, typename ... Args>
void Particle<Engine>::Boundary(int flag, TMaterialTag in, TMaterialTag out, Real dt, Args const &... args)
{

	UNIMPLEMENT;

//	auto selector = mesh.tags().template BoundarySelector<VERTEX>(in, out);
//
//// @NOTE: difficult to parallism
//
//	auto fun = [&](index_type idx)
//	{
//		if(!selector(idx)) return;
//
//		auto & cell = this->data_[idx];
//
//		auto pt = cell.begin();
//
//		while (pt != cell.end())
//		{
//			auto p = pt;
//			++pt;
//
//			index_type dest=idx;
//			if (flag == REFELECT)
//			{
//				coordinates_type x;
//
//				nTuple<3,Real> v;
//
//				Engine::InvertTrans(*p,&x,&v,std::forward<Args const &>(args)...);
//
//				dest=this->mesh.Refelect(idx,dt,&x,&v);
//
//				Engine::Trans(x,v,&(*p),std::forward<Args const &>(args)...);
//			}
//
//			if (dest != idx)
//			{
//				data_[dest].splice(data_[dest].begin(), cell, p);
//			}
//			else
//			{
//				cell.erase(p);
//			}
//
//		}
//	};
//
//// @NOTE: is a simple implement, need  parallism
//
//	mesh.SerialTraversal(VERTEX, fun);

}
template<class Engine> template<typename ... Args>
void Particle<Engine>::Collide(Args const &... args)
{
	UNIMPLEMENT;
}

//******************************************************************************************************
template<typename TX, typename TV, typename FE, typename FB> inline
void BorisMethod(Real dt, Real cmr, FE const & fE, FB const &fB, TX *x, TV *v)
{
// @ref  Birdsall(1991)   p.62
// Bories Method

	(*x) += (*v) * 0.5 * dt;

	auto B = real(fB((*x)));
	auto E = real(fE((*x)));

	Vec3 v_;

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
