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
		pic_type<Engine::Point_s>>
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

	template<typename ...Args>
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
						engine_type::mesh_.MakeContainer(GEOMETRY_TYPE,
								cell_type(
										container_type::begin()->get_allocator()))));

		engine_type::mesh_.ForAllCell(

		[&](index_type const & s)
		{
			auto & cell=this->operator[](s);
			auto pt = cell.begin();
			while (pt != cell.end())
			{
				auto p = pt;
				++pt;

				auto j = engine_type::mesh_.SearchCell(s,p->x);

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
		ForAllParticle<void(particle_type &, Real, Real, Args...)>(fun, m_, q_,
				std::forward<Args>(args) ...);
	}

	template<typename TFUN, typename TJ, typename ... Args>
	inline void ForEach(TFUN const & fun, TJ & J, Args const & ... args) const
	{
		engine_.mesh_.ForAllCell(

		[&](index_type const &s)
		{
			auto packs=PackParameters(MakeCache(J,s),MakeCache(args,s)...);
			for()

		});
	}

	template<typename ... Args>
	inline void Push(Args const& ... args)
	{
		Push(MakeCache(args) ...);
	}

	template<typename ... Args>
	inline void Push(RWCache<Args> const& ... args)
	{

		engine_.mesh_.ForAllCell(

		[&](index_type const &s,RWCache<Args> const& ...args)
		{
			ResetCache(args...);
			for(auto p:this->operator[](s))
			{
				engine_.Push(p,args...);
			}

		}

		);
	}

	template<typename TFUN, typename TJ, typename ... Args>
	inline void Scatter(TFUN const & fun, TJ & J, Args const & ... args) const
	{
		ForAllParticle(fun, J, args ...);
	}

	template<typename TJ, typename ... Args>
	inline void ScatterJ(TJ & J, Args const & ... args) const
	{
		ForAllParticle<
				void(particle_type const&, Real, Real, TJ &, Args const &...)>(
				engine_type::ScatterJ, J, args ...);
	}

	template<typename TN, typename ... Args>
	inline void ScatterN(TN & n, Args & ... args) const
	{
		ForAllParticle<
				void(particle_type const&, Real, Real, TN &, Args const &...)>(
				engine_type::ScatterJ, m_, q_, n, args ...);
	}

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
		engine_type::mesh_.ForAll(GEOMETRY_TYPE,

		[&](index_type const & s)
		{
			ForParticlesInCell(this->operator[](s),fun, args...);
		}

		);
	}
	template<typename Fun, typename ...Args>
	void ForAllParticle(Fun const & fun, Args &... args) const
	{
		engine_type::mesh_.ForAll(GEOMETRY_TYPE,

		[&](index_type const & s)
		{
			ForParticlesInCell(this->operator[](s),fun, args...);
		}

		);

	}

private:

	template<typename TCELL, typename Fun, typename ... Args>
	void ForParticlesInCell(TCELL & cell, Fun const & fun, Args & ... args)
	{
		for (auto & p : cell)
		{
			fun(p, args...);
		}
	}

	template<typename TCELL, typename Fun, typename ... Args>
	void ForParticlesInCell(TCELL const& cell, Fun const & fun,
			Args & ... args) const
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
