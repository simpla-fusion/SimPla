/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <cstddef>
#include <list>
#include <map>
#include <string>

#include "../fetl/fetl.h"
#include "../utilities/lua_state.h"
//need  libstdc++
//#include <ext/mt_allocator.h>
//#include <bits/allocator.h>
//template<typename T> using FixedSmallObjectAllocator=std::allocator<T>;
//		__gnu_cxx::__mt_alloc<T>;

namespace simpla
{
template<typename, typename > struct Field;
template<typename, int> struct Geometry;
struct LuaObject;

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
	inline void Push(Args const & ... args)
	{
		_Push(std::forward<Args const &>(args)...);

	}
	template<int N, typename TJ, typename ... Args>
	inline void Collect(TJ *J, Args const &... args)
	{
		_Collect(Int2Type<N>(), J, std::forward<Args const &>(args)...);
	}

	virtual std::string TypeName()
	{
		return "UNNAMED";
	}

	virtual std::ostream & Serialize(std::ostream & os) const
	{
		return os;
	}

private:

	//========================================================================
	// interface
	typedef typename TM::scalar scalar;
	DEFINE_FIELDS (mesh_type)

	virtual void _Push(Form<1> const &, Form<2> const &)
	{
	}

	virtual void _Push(VectorForm<0> const &, VectorForm<0> const &)
	{
	}

#define DEF_COLLECT_INTERFACE( _N_ ,_TJ_,_M_)																\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> &, Form<1> const &,	Form<2> const &) {}				\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> &, VectorForm<0> const & ,	VectorForm<0> const &) {}		\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> &) {}

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
class Particle:

public Engine,

		public Engine::mesh_type::template Container<
				std::list<typename Engine::Point_s> >,

		public ParticleBase<typename Engine::mesh_type>
{
	static const int GEOMETRY_TYPE = 0;

	std::list<typename Engine::Point_s> pool_;

public:

	typedef Engine engine_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::Point_s particle_type;

	typedef particle_type value_type;

	typedef typename mesh_type::scalar scalar;

	//mesh

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::index_type index_type;

	//container

	typedef std::list<typename Engine::Point_s> cell_type;

	typedef typename cell_type::allocator_type allocator_type;

	typedef typename mesh_type::template Container<
			std::list<typename Engine::Point_s> > container_type;

	mesh_type const &mesh;

public:

	template<typename ...Args>
	Particle(mesh_type const & pmesh) :

			engine_type(pmesh),

			container_type(
					std::move(
							pmesh.template MakeContainer<GEOMETRY_TYPE,
									cell_type>())),

			mesh(pmesh)

	{
	}

	virtual ~Particle()
	{
	}

	virtual std::string TypeName()
	{
		return engine_type::TypeName();
	}

	template<typename PT>
	inline void Deserialize(PT const &vm)
	{
		engine_type::Deserialize(vm);

		size_t num_pic;

		vm.template GetValue<size_t>("PIC", &num_pic);

		value_type default_value = engine_type::DefaultValue();

		ResizeCells(num_pic, default_value);
	}

	template<typename PT>
	inline void Serialize(PT &vm) const
	{
		engine_type::Serialize(vm);
	}

	std::ostream & Serialize(std::ostream & os) const
	{
		engine_type::Serialize(os);
		return os;
	}

	inline void ResizeCells(size_t num_pic,
			particle_type const & default_value = particle_type())
	{
		for (auto & cell : *this)
		{
			cell.resize(num_pic, default_value);
		}

	}

	void Sort()
	{

		container_type tmp(
				std::move(
						mesh.MakeContainer(GEOMETRY_TYPE,
								cell_type(
										container_type::begin()->get_allocator()))));

		mesh.ForAllCell(

		[&](index_type const & s)
		{
			auto & cell=this->operator[](s);
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
						tmp.at(j).splice(container_type::at(j).end(), cell, p);
					}
					catch (...)
					{
						cell.erase(p);
					}
				}
			}
		}

		);
		auto it1 = container_type::begin();
		auto it2 = tmp.begin();
		for (; it1 != container_type::end(); ++it1, ++it2)
		{
			it1->splice(it1->begin(), *it2);
		}
	}

	template<typename ... Args>
	inline void Push(Args const& ... args)
	{
		ForEachCell(

		[&](particle_type & p,
				typename ProxyCache<const Args>::type const& ... args_c)
		{
			engine_type::Push(p,args_c...);
		},

		args...);
	}

	template<int I, typename TJ, typename ... Args>
	inline void Collect(TJ & J, Args const & ... args) const
	{
//		ForEachCell(
//
//		[&](particle_type const& p,typename ProxyCache<TJ>::type & J_c,
//				typename ProxyCache<const Args>::type const& ... args_c)
//		{
//			engine_type::Collect(Int2Type<I>(),p,J_c,args_c...);
//		},
//
//		J, args...);
	}

	template<typename TFun, typename ... Args>
	inline void Function(TFun &fun, Args const& ... args)
	{
		ForEachCell(

		[&](particle_type & p,
				typename ProxyCache<const Args>::type const& ... args_c)
		{
			fun(p,args_c...);
		},

		args...);
	}

	/**
	 *  Traversal each cell, include boundary cells.
	 *
	 * @param fun (cell_type & cell,index_type const & s )
	 */

	template<typename Fun, typename ...Args>
	void ForEachCell(Fun const & fun, Args &... args)
	{
		/***
		 *  @BUG G++ Compiler bug (g++ <=4.8), need workaround.
		 *  Bug 41933 - [c++0x] lambdas and variadic templates don't work together
		 *   http://gcc.gnu.org/bugzilla/show_bug.cgi?id=41933
		 **/
//		mesh.ForAll(GEOMETRY_TYPE,
//
//		[&](index_type const & s)
//		{
//			ForParticlesInCell(this->operator[](s),
//					fun,ProxyCache< Args>::Eval(args,s)...);
//		}
//
//		);
	}
	template<typename Fun, typename ...Args>
	void ForEachCell(Fun const & fun, Args &... args) const
	{
//		mesh.ForAll(
//
//		[&](index_type const & s)
//		{
//			ForParticlesInCell(this->operator[](s),
//					fun, ProxyCache< Args>::Eval(args,s)...);
//		}
//
//		);

	}

private:

	template<typename TCELL, typename Fun, typename ... Args>
	void ForParticlesInCell(TCELL & cell, Fun & fun, Args && ... args)
	{
		for (auto & p : cell)
		{
			fun(p, args...);
		}
	}

	template<typename TCELL, typename Fun, typename ... Args>
	void ForParticlesInCell(TCELL const& cell, Fun & fun, Args &&... args) const
	{
		for (auto const& p : cell)
		{
			fun(p, args...);
		}
	}
private:

	//========================================================================
	// interface

	DEFINE_FIELDS (mesh_type)

	virtual void _Push(Form<1> const & E, Form<2> const &B)
	{
		Push(E, B);
	}

	virtual void _Push(VectorForm<0> const &E, VectorForm<0> const & B)
	{
		Push(E, B);
	}

#define DEF_COLLECT_INTERFACE( _N_ ,_TJ_,_M_)																\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> & J, Form<1> const & E,	Form<2> const & B)const {Collect<_N_>(J,E,B);}	\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> & J, VectorForm<0> const & E,	VectorForm<0> const & B)const {Collect<_N_>(J,E,B);}	\
	virtual void _Collect(Int2Type< _N_ >, _TJ_ <_M_> & J)const {Collect<_N_>(J);}

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

template<typename TParticleEngine>
std::shared_ptr<ParticleBase<typename TParticleEngine::mesh_type> > CreateParticle(
		typename TParticleEngine::mesh_type const & mesh, LuaObject const &cfg)
{

	typedef Particle<TParticleEngine> particle_type;
	typedef typename TParticleEngine::mesh_type mesh_type;

	return std::dynamic_pointer_cast<ParticleBase<mesh_type> >(
			std::shared_ptr<ParticleBase<mesh_type> >(new particle_type(mesh)));
}

template<typename TM>
class ParticleCollection: public std::map<std::string,
		std::shared_ptr<ParticleBase<TM> > >
{
public:
	typedef TM mesh_type;

	typedef ParticleBase<mesh_type> particle_type;

	typedef LuaObject configure_type;

	typedef std::map<std::string, std::shared_ptr<particle_type> > base_type;

	typedef std::function<
			std::shared_ptr<particle_type>(mesh_type const &,
					configure_type const &)> create_fun;

	typedef ParticleCollection<mesh_type> this_type;

private:
	std::map<std::string, create_fun> factory_;
public:

	mesh_type const & mesh;

	template<typename U>
	friend std::ostream & operator<<(std::ostream & os,
			ParticleCollection<U> const &self);

	ParticleCollection(mesh_type const & pmesh) :
			mesh(pmesh)
	{
	}
	~ParticleCollection()
	{
	}

	inline void Deserialize(configure_type const &cfg)
	{
		if (cfg.isNull())
			return;

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

			try
			{

				auto it = factory_.find(engine);

				if (it != factory_.end())
				{

					this->emplace(
							std::make_pair(key, it->second(mesh, p.second)));
				}

			} catch (...)
			{
				WARNING << "I do not know how to create \"" << key
						<< "\" particle! [engine=" << engine << "]";

				return;
			}

		}

	}

	void RegisterFactory(std::string const &engine_name, create_fun const &fun)
	{
		factory_.emplace(std::make_pair(engine_name, fun));
	}
	template<typename TEngine>
	void RegisterFactory(std::string const &engine_name)
	{
		factory_.emplace(std::make_pair(engine_name, &CreateParticle<TEngine>));
	}

	template<typename PT>
	inline void Serialize(PT &vm) const
	{
		WARNING << "UNIMPLEMENT!!";
	}

	template<typename ... Args>
	void PushAll(Args const & ... args)
	{
		for (auto & p : *this)
		{
			p.second->push(std::forward<Args const &>(args)...);
		}
	}

	template<typename TJ, typename ... Args>
	void CollectAll(TJ *J, Args const & ... args)
	{
		for (auto & p : *this)
		{
			p.second->Collect(J, std::forward<Args const &>(args)...);
		}
	}

};

template<typename TM>
std::ostream & operator<<(std::ostream & os, ParticleCollection<TM> const &self)
{
	os << "-- Particle Collection " << std::endl;
	os << "{";

	for (auto const & p : self)
	{
		os << p.first << " =  ";
		p.second->Serialize(os);

		os << ",";
	}
	os << "}" << std::endl;
	return os;

}

}
// namespace simpla

#endif /* PARTICLE_H_ */
