/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_
#include <cstddef>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <thread>

#include "../fetl/fetl.h"

#include "../utilities/log.h"
#include "../utilities/lua_state.h"
#include "../utilities/memory_pool.h"
#include "../utilities/type_utilites.h"

#ifndef NO_STD_CXX

//need  libstdc++

#include <ext/mt_allocator.h>
template<typename T> using FixedSmallSizeAlloc=__gnu_cxx::__mt_alloc<T>;

#endif
namespace simpla
{
struct LuaObject;

template<typename, typename > struct Field;

template<typename, int> struct Geometry;

template<typename TM> class ParticleBase;

//*******************************************************************************************************
template<class Engine>
class Particle: public Engine, public ParticleBase<typename Engine::mesh_type>
{

public:

	enum
	{
		IForm = 3
	};

	typedef Engine engine_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef typename engine_type::Point_s particle_type;

	typedef particle_type value_type;

	typedef typename mesh_type::scalar_type scalar_type;

	//mesh

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::index_type index_type;

	//container

	typedef std::list<value_type, FixedSmallSizeAlloc<value_type> > cell_type;

	typedef typename cell_type::allocator_type allocator_type;

	typedef std::vector<cell_type> container_type;

public:
	mesh_type const &mesh;

private:

	cell_type pool_;

	container_type data_;

	std::vector<container_type> mt_data_; // for sort

public:

	template<typename ...Args> Particle(mesh_type const & pmesh);

	virtual ~Particle();

	virtual std::string TypeName()
	{
		return engine_type::TypeName();
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

	void accept(VistorBase* vistor) const
	{
		vistor->visit(this);
	}

	void accept(VistorBase* vistor)
	{
		vistor->visit(this);
	}

	/**
	 *  Dump particles to a continue memory block
	 *  !!!this is a heavy operation!!!
	 *
	 * @return <datapoint , number of particles>
	 */
	std::pair<std::shared_ptr<value_type>, size_t> DumpData() const;

	allocator_type GetAllocator()
	{
		return pool_.get_allocator();
	}

	cell_type & operator[](size_t s)
	{
		return data_.at(s);
	}
	cell_type const & operator[](size_t s) const
	{
		return data_.at(s);
	}

	void Update();

	void Deserialize(LuaObject const &cfg);

	std::ostream & Serialize(std::ostream & os) const;

	template<typename ... Args> void NextTimeStep(Real dt, Args const& ... args);

	template<typename TJ, typename ... Args> void Collect(TJ * J, Args const & ... args) const;

	template<typename TFun, typename ... Args>
	inline void Function(TFun &fun, Args const& ... args)
	{
		_ForEachCell([&](particle_type & p,
				typename ProxyCache<const Args>::type const& ... args_c)
		{	fun(p,args_c...);}, args...);
	}

	template<typename ...Args> inline void Insert(size_t s, Args const & ...args)
	{
		data_[s].emplace_back(engine_type::Trans(std::forward<Args const &>(args)...));
	}

	void Sort();

};

template<class Engine>
template<typename ...Args> Particle<Engine>::Particle(mesh_type const & pmesh) :
		engine_type(pmesh), mesh(pmesh)
{
}

template<class Engine>
Particle<Engine>::~Particle()
{
}

template<class Engine>
void Particle<Engine>::Deserialize(LuaObject const &cfg)
{
	if (cfg.empty())
		return;

	Update();

	LOGGER

	<< "Particle:[ Name=" << cfg["Name"].as<std::string>()

	<< ", Engine=" << cfg["Engine"].as<std::string>() << "]";

	engine_type::Deserialize(cfg);

	size_t num_pic;

}
template<class Engine>
std::pair<std::shared_ptr<typename Engine::Point_s>, size_t> Particle<Engine>::DumpData() const
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
	os << "{ ";

	engine_type::Serialize(os)

//	<< Data(*this, engine_type::name_)

			;

	os << "} ";

	return os;
}

template<class Engine>
void Particle<Engine>::Update()
{
	if (data_.size() < mesh.GetNumOfElements(IForm))
		data_.resize(mesh.GetNumOfElements(IForm), cell_type(GetAllocator()));

	const unsigned int num_threads = std::thread::hardware_concurrency();

	if (mt_data_.size() < num_threads)
	{
		mt_data_.resize(num_threads);

		for (auto & d : mt_data_)
		{
			d.resize(mesh.GetNumOfElements(IForm), cell_type(GetAllocator()));
		}
	}
}

template<class Engine>
void Particle<Engine>::Sort()
{
	Update();

	const unsigned int num_threads = std::thread::hardware_concurrency();

	std::vector<std::thread> threads;

	auto fun = [this](unsigned int t_num,unsigned int t_id )
	{

		this->mesh._Traversal(t_num, t_id, this->IForm,

				[&](index_type const &src)
				{
					auto & cell=this->data_[src];
					auto pt = cell.begin();
					while (pt != cell.end())
					{
						auto p = pt;
						++pt;

						auto dest = this->mesh.SearchCell(src,p->x);

						if (dest==src)
						return;

						if(dest>0 && dest<mesh.GetNumOfElements(IForm))
						{
							this->mt_data_[t_id][dest].splice(this->mt_data_[t_id][dest].begin(), cell, p);
						}
						else
						{
							cell.erase(p);
						}
					}
				}

		);

	};

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		threads.emplace_back(std::thread(fun, num_threads, thread_id));
	}

	for (auto & t : threads)
	{
		t.join();
	}

	auto fun2 =

	[this](unsigned int t_num,unsigned int t_id)
	{
		this->mesh._Traversal(t_num, t_id, this->IForm,

				[&](index_type const &src)
				{
					for (int i = 0; i < t_num; ++i)
					{
						this->data_[src].splice(this->data_[src].begin(),this->mt_data_[i][src] );
					}
				},mesh_type::WITH_GHOSTS
		);

	};

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		threads[thread_id] = std::thread(fun, num_threads, thread_id);
	}

	for (auto & t : threads)
	{
		t.join();
	}

}

template<class Engine>
template<typename ...Args>
void Particle<Engine>::NextTimeStep(Real dt, Args const& ... args)
{
	if (data_.empty())
	{
		WARNING << "Particle [" << engine_type::name_ << "] is not initialized!";
		return;
	}
	LOGGER << "Move particle [" << engine_type::name_ << "]!";

	const unsigned int num_threads = std::thread::hardware_concurrency();

	std::vector<std::thread> threads;

	auto fun = [this,dt](unsigned int num_threads,unsigned int thread_id,
			typename ProxyCache<const Args>::type const& ... args_c)
	{

		this->mesh._Traversal(num_threads, thread_id, this->IForm,

				[&](index_type const &s)
				{
					for (auto & p : this->data_[s])
					{
						engine_type::NextTimeStep(&p,dt, args_c...);
					}
				}

		);

	};

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		threads.emplace_back(std::thread(fun, num_threads, thread_id,

		ProxyCache<Args const>::Eval(args)...));
	}

	for (auto & t : threads)
	{
		t.join();
	}

	Sort();

}

template<class Engine>
template<typename TJ, typename ...Args>
void Particle<Engine>::Collect(TJ * J, Args const & ... args) const
{
	if (data_.empty())
	{
		WARNING << "Particle [" << engine_type::name_ << "] is not initialized!";
		return;
	}

	LOGGER << "Collect particle [" << engine_type::name_ << "] to Form<" << TJ::IForm << ","
			<< (is_ntuple<typename TJ::value_type>::value ? "Vector" : "Scalar") << ">!";

	const unsigned int num_threads = std::thread::hardware_concurrency();

	std::vector<TJ> tmp(num_threads, *J);

	for (auto &v : tmp)
	{
		v.Fill(0);
	}

	std::vector<std::thread> threads;

	auto fun = [this](unsigned int t_num,unsigned int t_id,
			typename ProxyCache<TJ*>::type J_c,
			typename ProxyCache<const Args>::type ... args_c)
	{

		this->mesh._Traversal(t_num, t_id, this->IForm,

				[&](index_type const &s)
				{
					for (auto const& p : this->data_[s])
					{
						engine_type::Collect(p, J_c,args_c...);
					}
				},mesh_type::WITH_GHOSTS

		);

	};

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		threads.emplace_back(std::thread(fun, num_threads, thread_id,

		ProxyCache<TJ*>::Eval(&tmp[thread_id]),

		ProxyCache<Args const>::Eval(args)...

		));
	}

	for (auto & t : threads)
	{
		t.join();
	}

	for (int i = 0; i < num_threads; ++i)
	{
		*J += tmp[i];
	}
}

template<typename TM>
std::ostream & operator<<(std::ostream & os, Particle<TM> const &self)
{
	return self.Serialize(os);
}

//*******************************************************************************************************
template<typename TM>
struct PICEngineBase
{

protected:
	Real m_, q_;
	std::string name_;
public:
	typedef TM mesh_type;

public:

	mesh_type const &mesh;

	PICEngineBase(mesh_type const &pmesh) :
			mesh(pmesh), m_(1.0), q_(1.0), name_("unnamed")
	{

	}
	virtual ~PICEngineBase()
	{
	}

	std::string TypeName()
	{
		return _TypeName();
	}

	virtual std::string _TypeName() const
	{
		return "unknown";
	}

	inline Real GetMass() const
	{
		return m_;
	}

	inline Real GetCharge() const
	{
		return q_;
	}

	inline void SetMass(Real m)
	{
		m_ = m;
	}

	inline void SetCharge(Real q)
	{
		q_ = q;
	}

	const std::string& GetName() const
	{
		return name_;
	}

	void SetName(const std::string& name)
	{
		name_ = name;
	}

	virtual void Update()
	{
	}

	virtual void Deserialize(LuaObject const &vm)
	{
		vm.template GetValue("Mass", &m_);
		vm.template GetValue("Charge", &q_);
		vm.template GetValue("Name", &name_);
	}

	virtual std::ostream & Serialize(std::ostream & os) const
	{
		os

		<< "Name = " << name_ << " ,"

		<< "Mass = " << m_ << " , "

		<< "Charge = " << q_ << ","

		;

		return os;
	}

};

//*******************************************************************************************************

template<typename TParticleEngine>
std::shared_ptr<ParticleBase<typename TParticleEngine::mesh_type> > CreateParticle(
		typename TParticleEngine::mesh_type const & mesh)
{

	typedef Particle<TParticleEngine> particle_type;
	typedef typename TParticleEngine::mesh_type mesh_type;

	return std::dynamic_pointer_cast<ParticleBase<mesh_type> >(
			std::shared_ptr<ParticleBase<mesh_type> >(new particle_type(mesh)));
}

//*******************************************************************************************************

template<typename TM>
class ParticleBase
{

public:
	typedef TM mesh_type;

	ParticleBase()
	{

	}
	virtual ~ParticleBase()
	{

	}

//	virtual void accept(VistorBase*)=0;

	virtual std::string TypeName()
	{
		return "UNNAMED";
	}

	virtual void Deserialize(LuaObject const &cfg)
	{

	}

	virtual std::ostream & Serialize(std::ostream & os) const
	{
		return os;
	}

};

//*******************************************************************************************************
template<typename TM>
class ParticleCollection: public std::map<std::string, std::shared_ptr<ParticleBase<TM> > >
{
public:
	typedef TM mesh_type;

	typedef ParticleBase<mesh_type> particle_type;

	typedef LuaObject configure_type;

	typedef std::map<std::string, std::shared_ptr<particle_type> > base_type;

	typedef std::function<std::shared_ptr<particle_type>(mesh_type const &)> create_fun;

	typedef ParticleCollection<mesh_type> this_type;

private:
	std::map<std::string, create_fun> factory_;
public:

	mesh_type const & mesh;

	template<typename U>
	friend std::ostream & operator<<(std::ostream & os, ParticleCollection<U> const &self);

	ParticleCollection(mesh_type const & pmesh) :
			mesh(pmesh)
	{
	}
	~ParticleCollection()
	{
	}

	void RegisterFactory(std::string const &engine_name, create_fun const &fun)
	{
		factory_.emplace(engine_name, fun);
	}
	template<typename TEngine>
	void RegisterFactory(std::string engine_name = "")
	{
		if (engine_name == "")
			engine_name = TEngine::TypeName();

		RegisterFactory(engine_name, create_fun(&CreateParticle<TEngine>));
	}

	void Deserialize(configure_type const &cfg);

	std::ostream & Serialize(std::ostream & os) const;

	template<typename PT>
	inline void Serialize(PT &vm) const
	{
		WARNING << "UNIMPLEMENT!!";
	}

	template<typename ... Args> void NextTimeStep(Args const & ... args);

	template<typename TJ, typename ... Args> void Collect(TJ *J, Args const & ... args) const;
};

template<typename TM>
void ParticleCollection<TM>::Deserialize(configure_type const &cfg)
{
	if (cfg.empty())
		return;

	Logger logger(LOG_LOG);

	logger << "Load Particles " << endl << flush << indent;

	for (auto const &p : cfg)
	{
		std::string key;

		if (!p.first.is_number())
		{
			key = p.first.as<std::string>();
		}
		else
		{
			p.second.GetValue("Name", &key);
		}

		std::string engine = p.second.at("Engine").as<std::string>();

		auto it = factory_.find(engine);

		if (it != factory_.end())
		{
			auto t = it->second(mesh);

			t->Deserialize(p.second);

			this->emplace(key, t);
		}
		else
		{
			WARNING << "I do not know how to create \"" << key << "\" particle! [engine=" << engine << "]";

			return;
		}

	}

	logger << DONE;

}
template<typename TM>

std::ostream & ParticleCollection<TM>::Serialize(std::ostream & os) const
{

	os << "Particles={ \n";

	ContainerOutPut3(os, this->begin(), this->end(),

	[&](std::ostream & oos, decltype(this->begin()) const &it)->std::ostream &
	{
		return it->second->Serialize(oos)<<"\n";
	});

	os << "} \n";

	return os;
}

DEFINE_VISTOR (NextTimeStep);
DEFINE_VISTOR (Collect);
template<typename TM>
template<typename ... Args>
void ParticleCollection<TM>::NextTimeStep(Args const & ... args)
{
	for (auto & p : *this)
	{
		p.second->accept(CreateNexTimeStepVistor(std::forward<Args const &>(args)...));
	}
}
template<typename TM>
template<typename TJ, typename ... Args>
void ParticleCollection<TM>::Collect(TJ *J, Args const & ... args) const
{
	for (auto & p : *this)
	{
		p.second->accept(CreateCollectVistor(J,std::forward<Args const &>(args)...));
	}
}

template<typename TM>
std::ostream & operator<<(std::ostream & os, ParticleCollection<TM> const &self)
{
	return self.Serialize(os);
}

}
// namespace simpla

#endif /* PARTICLE_H_ */
