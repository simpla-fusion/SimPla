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

#include "../fetl/fetl.h"

#include "../utilities/log.h"
#include "../utilities/lua_state.h"
#include "../io/data_stream.h"
#include "load_particle.h"
#include "save_particle.h"

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

	static const int IForm = 0;

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

	typedef std::list<value_type

	, FixedSmallSizeAlloc<value_type>

	> cell_type;

	typedef typename cell_type::allocator_type allocator_type;

	typedef std::vector<cell_type> container_type;

public:
	mesh_type const &mesh;

private:

	cell_type pool_;

	container_type data_;

public:

	template<typename ...Args> Particle(mesh_type const & pmesh);

	virtual ~Particle();

	virtual std::string TypeName()
	{
		return engine_type::TypeName();
	}

	size_t size() const
	{
		return data_.size();
	}

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

	void Sort();

	void Deserialize(LuaObject const &cfg);

	std::ostream & Serialize(std::ostream & os) const;

	template<typename ... Args> void Push(Real dt, Args const& ... args);

	template<int I, typename TJ, typename ... Args> void Collect(TJ * J, Args const & ... args) const;

	template<typename TFun, typename ... Args>
	inline void Function(TFun &fun, Args const& ... args)
	{
		_ForEachCell([&](particle_type & p,
				typename ProxyCache<const Args>::type const& ... args_c)
		{	fun(p,args_c...);}, args...);
	}

private:

	/**
	 *  Traversal each cell, include boundary cells.
	 *
	 * @param fun (cell_type & cell,index_type const & s )
	 */

	template<typename Fun, typename ...Args>
	void _ForEachCell(Fun const & fun, Args &... args)
	{
		/***
		 *  @BUG G++ Compiler bug (g++ <=4.8), need workaround.
		 *  Bug 41933 - [c++0x] lambdas and variadic templates don't work together
		 *   http://gcc.gnu.org/bugzilla/show_bug.cgi?id=41933
		 **/
		mesh.TraversalIndex(IForm,

		[&](index_type const & s)
		{
			_ForParticlesInCell(data_[s], fun,
					ProxyCache< Args>::Eval(args,s)...);

		});
	}

	template<typename Fun, typename ...Args>
	void _ForEachCell(Fun const & fun, Args &... args) const
	{
		mesh.TraversalIndex(IForm, [&](index_type const & s)
		{
			_ForParticlesInCell(data_[s], fun,
					ProxyCache< Args>::Eval(args,s)...);
		});

	}

	template<typename TCELL, typename Fun, typename ... Args>
	void _ForParticlesInCell(TCELL & cell, Fun & fun, Args && ... args)
	{
		for (auto & p : cell)
		{
			fun(p, args...);
		}
	}

	template<typename TCELL, typename Fun, typename ... Args>
	void _ForParticlesInCell(TCELL const& cell, Fun & fun, Args &&... args) const
	{
		for (auto const& p : cell)
		{
			fun(p, args...);
		}
	}

	//========================================================================
	// interface

	DEFINE_FIELDS (mesh_type)

	virtual void _Push(Real dt, Form<1> const & E, Form<2> const &B)
	{
		Push(dt, E, B);
	}

	virtual void _Push(Real dt, VectorForm<0> const &E, VectorForm<0> const & B)
	{
		Push(dt, E, B);
	}

#define DEF_COLLECT_INTERFACE( _N_ ,_TJ_,_M_)																\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> * J, Form<1> const & E,	Form<2> const & B)const {Collect<_N_>(J,E,B);}	\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> * J, VectorForm<0> const & E,	VectorForm<0> const & B)const {Collect<_N_>(J,E,B);}	\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> * J)const {Collect<_N_>(J);}

	DEF_COLLECT_INTERFACE(0 , Form, 0 )
	DEF_COLLECT_INTERFACE(0 , Form, 3 )
	DEF_COLLECT_INTERFACE(1, VectorForm , 0 )
	DEF_COLLECT_INTERFACE(1, Form , 1 )
	DEF_COLLECT_INTERFACE(1, Form , 2 )
	DEF_COLLECT_INTERFACE(1, VectorForm , 3 )
	DEF_COLLECT_INTERFACE(2, TensorForm , 0 )
	DEF_COLLECT_INTERFACE(2, VectorForm , 1 )
	DEF_COLLECT_INTERFACE(2, VectorForm , 2 )
	DEF_COLLECT_INTERFACE(2, TensorForm , 3 )

#undef DEF_COLLECT_INTERFACE
};

template<class Engine>
template<typename ...Args> Particle<Engine>::Particle(mesh_type const & pmesh)
		: engine_type(pmesh), mesh(pmesh)
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

	data_.resize(mesh.GetNumOfElements(3));

	LOGGER

	<< "Particle:[ Name=" << cfg["Name"].as<std::string>()

	<< ", Engine=" << cfg["Engine"].as<std::string>() << "]";

	engine_type::Deserialize(cfg);

	size_t num_pic;

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
	data_.resize(mesh.GetNumOfElements(IForm), cell_type(GetAllocator()));
}

template<class Engine>
void Particle<Engine>::Sort()
{

	container_type tmp;

	tmp.resize(mesh.GetNumOfElements(IForm), cell_type(GetAllocator()));

	mesh.TraversalIndex(IForm,

	[&](index_type const & s)
	{
		auto & cell=data_[s];
		auto pt = cell.begin();
		while (pt != cell.end())
		{
			auto p = pt;
			++pt;

			auto j = mesh.SearchCell(s,p->x);

			if (j!=s)
			{
				try
				{
					tmp.at(j).splice(data_.at(j).end(), cell, p);
				}
				catch (...)
				{
					cell.erase(p);
				}
			}
		}
	}

	);
	auto it1 = data_.begin();
	auto it2 = tmp.begin();
	for (; it1 != data_.end(); ++it1, ++it2)
	{
		it1->splice(it1->begin(), *it2);
	}
}

template<class Engine>
template<typename ...Args>
void Particle<Engine>::Push(Real dt, Args const& ... args)
{
	if (data_.empty())
	{
		WARNING << "Particle [" << engine_type::name_ << "] is not initialized!";
		return;
	}
	LOGGER << "Push particle [" << engine_type::name_ << "]!";

	_ForEachCell(

	[&](particle_type & p, typename ProxyCache<const Args>::type const& ... args_c)
	{
		engine_type::Push(p,dt,args_c...);
	},

	args...);

	Sort();

}

template<class Engine>
template<int I, typename TJ, typename ...Args>
void Particle<Engine>::Collect(TJ * J, Args const & ... args) const
{
	if (data_.empty())
	{
		WARNING << "Particle [" << engine_type::name_ << "] is not initialized!";
		return;
	}

	LOGGER << "Collect particle [" << engine_type::name_ << "] to Form<" << I << ","
	        << (is_ntuple<typename TJ::value_type>::value ? "Vector" : "Scalar") << ">!";

	_ForEachCell(

	[&](particle_type const& p,typename ProxyCache<TJ>::type & J_c,
			typename ProxyCache<const Args>::type const& ... args_c)
	{
		engine_type::Collect(Int2Type<I>(),p,&J_c,args_c...);

	}, *J, args...);
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

	PICEngineBase(mesh_type const &pmesh)
			: mesh(pmesh), m_(1.0), q_(1.0), name_("unnamed")
	{

	}
	~PICEngineBase()
	{
	}

	virtual std::string TypeName() const
	{
		return "Default";
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

	template<typename ... Args>
	inline void Push(Real dt, Args const & ... args)
	{
		_Push(dt, std::forward<Args const &>(args)...);

	}
	template<int N, typename TJ, typename ... Args>
	inline void Collect(TJ *J, Args const &... args) const
	{
		_Collect(Int2Type<N>(), J, std::forward<Args const &>(args)...);
	}

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

private:

	//========================================================================
	// interface
	typedef typename TM::scalar scalar;DEFINE_FIELDS (mesh_type)

	virtual void _Push(Real dt, Form<1> const &, Form<2> const &)
	{
		UNIMPLEMENT << " Particle Push operation";
	}

	virtual void _Push(Real dt, VectorForm<0> const &, VectorForm<0> const &)
	{
		UNIMPLEMENT << " Particle Push operation";
	}

#define DEF_COLLECT_INTERFACE( _N_ ,_TJ_,_M_)																\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> *, Form<1> const &,Form<2> const &)const {}			\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> *, VectorForm<0> const & ,VectorForm<0> const &)const {}\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> *)const {}

	DEF_COLLECT_INTERFACE(0 , Form, 0 )
	DEF_COLLECT_INTERFACE(0 , Form, 3 )
	DEF_COLLECT_INTERFACE(1, VectorForm , 0 )
	DEF_COLLECT_INTERFACE(1, Form , 1 )
	DEF_COLLECT_INTERFACE(1, Form , 2 )
	DEF_COLLECT_INTERFACE(1, VectorForm , 3 )
	DEF_COLLECT_INTERFACE(2, TensorForm , 0 )
	DEF_COLLECT_INTERFACE(2, VectorForm , 1 )
	DEF_COLLECT_INTERFACE(2, VectorForm , 2 )
	DEF_COLLECT_INTERFACE(2, TensorForm , 3 )

#undef DEF_COLLECT_INTERFACE

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

	ParticleCollection(mesh_type const & pmesh)
			: mesh(pmesh)
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
	void RegisterFactory(std::string const &engine_name)
	{
		RegisterFactory(engine_name, create_fun(&CreateParticle<TEngine>));
	}

	void Deserialize(configure_type const &cfg);

	std::ostream & Serialize(std::ostream & os) const;

	template<typename PT>
	inline void Serialize(PT &vm) const
	{
		WARNING << "UNIMPLEMENT!!";
	}

	template<typename ... Args>
	void NextTimeStep(Args const & ... args)
	{
		for (auto & p : *this)
		{
			p.second->Push(std::forward<Args const &>(args)...);
		}
	}

	template<typename TJ, typename ... Args>
	void Collect(TJ *J, Args const & ... args) const
	{
		for (auto & p : *this)
		{
			p.second->template Collect<TJ::IForm>(J, std::forward<Args const &>(args)...);
		}
	}

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

template<typename TM>
std::ostream & operator<<(std::ostream & os, ParticleCollection<TM> const &self)
{
	return self.Serialize(os);
}

}
// namespace simpla

#endif /* PARTICLE_H_ */
