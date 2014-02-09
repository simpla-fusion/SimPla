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
template<typename, typename > struct Field;

template<typename, int> struct Geometry;

//*******************************************************************************************************
template<class Engine>
class Particle: public Engine
{

public:

	enum
	{
		IForm = 3
	};

	typedef Engine engine_type;

	typedef ParticleBase<typename Engine::mesh_type> base_type;

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

	template<typename ...Args> Particle(mesh_type const & pmesh, std::string const & n = "");

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

	template<typename ...Args> inline void Insert(size_t s, Args const & ...args)
	{
		data_[s].emplace_back(engine_type::Trans(std::forward<Args const &>(args)...));
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

	void Update();

	void DumpData(std::string const &path);

	void Deserialize(LuaObject const &cfg);

	std::ostream & Serialize(std::ostream & os) const;

	//***************************************************************************************************
	template<typename TFun, typename ... Args>
	void Function(TFun &fun, Args const& ... args) const;

	template<typename TFun, typename ... Args>
	void Function(TFun &fun, Args const& ... args);

	template<typename ... Args> void NextTimeStep(Real dt, Args const& ... args);

	template<typename TJ, typename ... Args> void Collect(TJ * J, Args const & ... args) const;

	void Sort();

	template<typename ... Args>
	void Boundary(int flag, typename mesh_type::tag_type in, typename mesh_type::tag_type out, Real dt,
	        Args const& ... args);

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

	Real GetClock() const
	{
		return clock_;
	}

	void SetClock(Real clock)
	{
		clock_ = clock;
	}

	std::string const &GetName() const
	{
		return name_;
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

	Real clock_;

	std::string name_;

};

template<class Engine>
template<typename ...Args> Particle<Engine>::Particle(mesh_type const & pmesh, std::string const & n)
		: engine_type(pmesh), mesh(pmesh), isSorted_(true), clock_(0), name_(n)
{
}

template<class Engine>
Particle<Engine>::~Particle()
{
}

template<class Engine>
void Particle<Engine>::Deserialize(LuaObject const &cfg)
{
	Update();

	LoadParticle(cfg, this);

	base_type::Deserialize(cfg);

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
std::ostream & Particle<Engine>::Serialize(std::ostream & os) const
{
	os << "{ Name = '" << base_type::GetName() << "' , ";

	engine_type::Serialize(os);

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
void Particle<Engine>::Update()
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

	if (trace_cache_.empty())
	{
		size_t num_particle = trace_particle_.size();
		trace_cache_.resize(num_particle * trace_cache_depth_);
		trace_tail_ = trace_cache_.begin();
	}

	for (auto const& p : trace_particle_)
	{
		*trace_tail_ = *p;
		++trace_tail_;
	}

	if (trace_tail_ == trace_cache_.end())
	{

		size_t dims[2] =
		{	trace_particle_.size(), trace_cache_.size() / trace_particle_.size()};

		LOGGER << Dump(&trace_cache_[0], base_type::GetName(), 2, dims, true);

		trace_tail_ = trace_cache_.begin();
	}
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

	Update();

	if (base_type::IsSorted())
		return;

	const unsigned int num_threads = std::thread::hardware_concurrency();

	try
	{

		std::vector<std::thread> threads;

		for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
		{
			threads.emplace_back(std::thread(

			[this](unsigned int t_num,unsigned int t_id )
			{
				this->mesh._Traversal(t_num, t_id, this->IForm,
						[this,t_id](index_type const &src)
						{	Resort(src,&(this->mt_data_[t_id]));});
			}

			, num_threads, thread_id)

			);
		}

		for (auto & t : threads)
		{
			t.join();
		}

	} catch (std::exception const & e)
	{
		ERROR << e.what();

	}

	try
	{
		std::vector<std::thread> threads2;

		for (int thread_id = 0; thread_id < num_threads; ++thread_id)
		{
			threads2.emplace_back(std::thread(

			[this]( int t_num, int t_id)
			{
				this->mesh._Traversal(t_num, t_id, this->IForm,

						[&](index_type const &s)
						{
							for (int i = 0; i < t_num; ++i)
							{
								this->data_[s].splice(this->data_[s].begin(),this->mt_data_[i][s] );
							}
						}
				);

			}

			, num_threads, thread_id));
		}

		for (auto & t : threads2)
		{
			t.join();
		}
	} catch (std::exception const & e)
	{
		ERROR << e.what();
	}

	isSorted_ = true;
}

template<class Engine>
template<typename ...Args>
void Particle<Engine>::NextTimeStep(Real dt, Args const& ... args)
{
	if (data_.empty())
	{
		WARNING << "Particle [" << base_type::GetName() << " : " << engine_type::GetTypeAsString()
		        << "] is not initialized!";
		return;
	}

	Sort();

	base_type::NextTimeStep(dt, std::forward<Args const&>(args) ...);

	const unsigned int num_threads = std::thread::hardware_concurrency();

	std::vector<std::thread> threads;

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		threads.emplace_back(std::thread(

		[this,num_threads, thread_id,dt]( Cache<const Args> ... args_c2)
		{
			this->mesh._Traversal(num_threads, thread_id, this->IForm,

					[&](index_type const &s)
					{

						for (auto & p : this->data_[s])
						{
							RefreshCache(s,args_c2...);

							engine_type::NextTimeStep(&p, dt, *args_c2...);
						}
					}
			);

		},

		Cache<const Args >(args , engine_type::GetAffectedRegion())...));
	}

	for (auto & t : threads)
	{
		t.join();
	}

	isSorted_ = false;

	clock_ += dt;

	Sort();

}

template<class Engine>
template<typename TJ, typename ...Args>
void Particle<Engine>::Collect(TJ * J, Args const & ... args) const
{
	if (data_.empty())
	{
		WARNING << "Particle [" << base_type::GetName() << " : " << engine_type::GetTypeAsString()
		        << "] is not initialized!";
		return;
	}

	if (!base_type::IsSorted())
		ERROR << "Particles are not sorted!";

	const unsigned int num_threads = std::thread::hardware_concurrency();

	std::vector<std::thread> threads;

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{

		/***
		 *  @NOTICE std::thread  accept parameter by VALUE!!!
		 *     NOT by REFERENCE!!!!
		 */

		threads.emplace_back(

		std::thread(

		[this,num_threads, thread_id]( Cache<TJ*> J_c2, Cache<const Args> ... args_c2)
		{

			this->mesh._Traversal(num_threads, thread_id, this->IForm,

					[&](index_type const &s)
					{
						RefreshCache(s,J_c2,args_c2...);

						for (auto const& p : this->data_[s])
						{
							engine_type::Collect(p, &(*J_c2) , *args_c2...);
						}

						FlushCache(J_c2,args_c2...);

					}
			);
		}

		, Cache<TJ*>(J, engine_type::GetAffectedRegion())

		, Cache<const Args >(args ,engine_type::GetAffectedRegion())...)

		);
	}

	for (auto & t : threads)
	{
		t.join();
	}

}

template<class Engine>
template<typename TFun, typename ...Args>
void Particle<Engine>::Function(TFun &fun, Args const& ... args)
{
	if (data_.empty())
	{
		WARNING << "Particle [" << engine_type::name_ << "] is not initialized!";
		return;
	}

	if (!base_type::IsSorted())
		ERROR << "Particles are not sorted!";

	const unsigned int num_threads = std::thread::hardware_concurrency();

	std::vector<std::thread> threads;

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		threads.emplace_back(

		std::thread(

		[this,num_threads, thread_id]( Cache<const Args>&& ... args_c2)
		{

			this->mesh._Traversal(num_threads, thread_id, this->IForm,

					[&](index_type const &s)
					{

						RefreshCache(s, args_c2...);

						for (auto const& p : this->data_[s])
						{
							fun(p, *args_c2...);
						}

					}
			);
		}

		, Cache<const Args >(args ,engine_type::GetAffectedRegion())...)

		);
	}

	for (auto & t : threads)
	{
		t.join();
	}

}

template<class Engine>
template<typename ... Args>
void Particle<Engine>::Boundary(int flag, typename mesh_type::tag_type in, typename mesh_type::tag_type out, Real dt,
        Args const &... args)
{
	auto selector = mesh.tags().template BoundarySelector<VERTEX>(in, out);

// @NOTE: difficult to parallism

	auto fun = [&](index_type src)
	{
		if(!selector(src)) return;

		auto & cell = this->data_[src];

		auto pt = cell.begin();

		while (pt != cell.end())
		{
			auto p = pt;
			++pt;

			index_type dest=src;
			if (flag == base_type::REFELECT)
			{
				coordinates_type x;

				nTuple<3,Real> v;

				Engine::InvertTrans(*p,&x,&v,std::forward<Args const &>(args)...);

				dest=this->mesh.Refelect(src,dt,&x,&v);

				Engine::Trans(x,v,&(*p),std::forward<Args const &>(args)...);
			}

			if (dest != src)
			{
				data_[dest].splice(data_[dest].begin(), cell, p);
			}
			else
			{
				cell.erase(p);
			}

		}
	};

// @NOTE: is a simple implement, need  parallism

	mesh.SerialTraversal(VERTEX, fun);

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
