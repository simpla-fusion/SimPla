/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <fetl/primitives.h>
#include <cstddef>
#include <fetl/field_rw_cache.h>
#include <include/simpla_defs.h>
#include <cstddef>
#include <list>
namespace simpla
{
template<typename T> using pic_type =std::list<T, FixedSmallObjectAllocator<T>>;

template<class Engine>
class Particle: public Engine, public Engine::mesh_type::template Container<
		pic_type<typename Engine::Point_s>>
{
	static const int GEOMETRY_TYPE = 0;

public:

	typedef Engine engine_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::Point_s particle_type;

	typedef particle_type value_type;

	//mesh

	typedef typename engine_type::mesh_type mesh_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::index_type index_type;

	//container
	typedef FixedSmallObjectAllocator<value_type> allocator_type;

	typedef std::list<value_type, allocator_type> cell_type;

	typedef typename mesh_type::template Container<cell_type> container_type;

	Particle(mesh_type const & mesh, allocator_type allocator =
			allocator_type()) :

			engine_type(mesh),

			container_type(
					std::move(
							mesh.template MakeContainer<cell_type>(
									GEOMETRY_TYPE, cell_type(allocator))))

	{
	}

	void Init(size_t num_pic)
	{
		value_type default_value;

		engine_type::SetDefaultValue(default_value);

		ResizeCells(num_pic, default_value);
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
						engine_type::mesh.MakeContainer(GEOMETRY_TYPE,
								cell_type(
										container_type::begin()->get_allocator()))));

		engine_type::mesh.ForAllCell(

		[&](index_type const & s)
		{
			auto & cell=this->operator[](s);
			auto pt = cell.begin();
			while (pt != cell.end())
			{
				auto p = pt;
				++pt;

				auto j = engine_type::mesh.SearchCell(s,p->x);

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

	template<typename TFUN, typename ... Args>
	inline void ForEach(TFUN const & fun, Args const& ... args)
	{
//		ForAllParticle<void(particle_type &, Real, Real, Args...)>(fun, m_, q_,
//				std::forward<Args>(args) ...);
	}

	template<typename TFUN, typename TJ, typename ... Args>
	inline void ForEach(TFUN const & fun, TJ & J, Args const & ... args) const
	{
//		engine_type::mesh.ForAllCell(
//
//		[&](index_type const &s)
//		{
//			auto packs=PackParameters(MakeCache(J,s),MakeCache(args,s)...);
//			for()
//
//		});
	}

	template<typename ... Args>
	inline void Push(Args const& ... args)
	{
		ForAllParticle(

		[&](particle_type & p,
				typename CacheProxy<const Args>::type const& ... args_c)
		{
			engine_type::Push(p,args_c...);
		},

		args...);
	}

	template<typename TFUN, typename TJ, typename ... Args>
	inline void Scatter(TFUN const & fun, TJ & J, Args const & ... args) const
	{

//		ForAllParticle(fun, J, args ...);
	}

	template<typename TJ, typename ... Args>
	inline void ScatterJ(TJ & J, Args const & ... args) const
	{
		ForAllParticle(

				[&](particle_type const& p,typename CacheProxy<TJ>::type const & J_c,
						typename CacheProxy<const Args>::type const& ... args_c)
				{
					engine_type::ScatterJ(p,J_c,args_c...);
				},

				J, args...);
	}

//	template<typename TN, typename ... Args>
//	inline void ScatterN(TN & n, Args & ... args) const
//	{
//		ForAllParticle(
//
//				[&](particle_type const& p,typename CacheProxy<TN>::type const & n_c,
//						typename CacheProxy<const Args>::type const& ... args_c)
//				{
//					engine_type::ScatterN(p,n_c,args_c...);
//				},
//
//				n, args...);
//	}

	/**
	 *  Traversal each cell, include boundary cells.
	 *
	 * @param fun (cell_type & cell,index_type const & s )
	 */

	template<typename Fun, typename ...Args>
	void ForAllParticle(Fun const & fun, Args &... args)
	{
		/***
		 *  @BUG G++ Compiler bug (g++ <=4.8), need workaround.
		 *  Bug 41933 - [c++0x] lambdas and variadic templates don't work together
		 *   http://gcc.gnu.org/bugzilla/show_bug.cgi?id=41933
		 **/
		engine_type::mesh.ForAll(GEOMETRY_TYPE,

		[&](index_type const & s)
		{
			ForParticlesInCell(this->operator[](s),
					fun,CacheProxy< Args>::Eval(args,s)...);
		}

		);
	}
	template<typename Fun, typename ...Args>
	void ForAllParticle(Fun const & fun, Args &... args) const
	{
		engine_type::mesh.ForAll(GEOMETRY_TYPE,

		[&](index_type const & s)
		{
			ForParticlesInCell(this->operator[](s),
					fun, CacheProxy< Args>::Eval(args,s)...);
		}

		);

	}

private:

	template<typename TCELL, typename Fun, typename ... Args>
	void ForParticlesInCell(TCELL & cell, Fun & fun, Args const& ... args)
	{
		for (auto & p : cell)
		{
			fun(p, args...);
		}
	}

	template<typename TCELL, typename Fun, typename ... Args>
	void ForParticlesInCell(TCELL const& cell, Fun & fun,
			Args const& ... args) const
	{
		for (auto const& p : cell)
		{
			fun(p, args...);
		}
	}

};

}
// namespace simpla

#endif /* PARTICLE_H_ */
