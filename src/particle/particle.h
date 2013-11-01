/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <ext/mt_allocator.h>
#include <fetl/geometry.h>
#include <fetl/primitives.h>
#include <include/simpla_defs.h>
#include <cstddef>
#include <list>

namespace simpla
{
template<typename TM, typename TV> class Particle;

template<typename T> using PIC=std::list<T, FixedSmallObjectAllocator<T> >;

template<typename TM, typename T>
class ParticleBase: public TM::Container<PIC<T> >
{
	TM const & mesh_;
	static const int GEOMETRY_TYPE = 0;
public:

	typename TM Mesh;

	typedef typename PIC<T>::allocator_type allocator_type;

	typedef T value_type;

	typedef T particle_type;

	typedef PIC<value_type> cell_type;

	typedef typename TM::Container<PIC<T> > container_type;

	typedef ParticleBase<Geometry, value_type> this_type;

	ParticleBase(TM const & m, allocator_type allocator = allocator_type()) :

			container_type(
					std::move(
							m.MakeContianer(GEOMETRY_TYPE,
									cell_type(allocator)))),

			allocator_(allocator), mesh_(m)
	{
	}

	ParticleBase(this_type const & r) :
			container_type(r), mesh_(r.mesh_)
	{
	}

	ParticleBase(this_type && r) :
			container_type(std::move(r)), mesh_(r.mesh_)
	{
	}

	~ParticleBase() = default;

	void Sort()
	{

		container_type tmp(
				std::move(
						mesh_.MakeContainer(GEOMETRY_TYPE,
								pic_type(allocator))))
		);

		ForAllCell(

		[&](index_type const & s)
		{
			auto & cell= this->operator[](s);

			auto pt = cell.begin();
			while (pt != cell.end())
			{
				auto p = it;
				++it;

				auto j = mesh_.SearchCell(s,p->x);

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
		});
		auto it1 = container_type::begin();
		auto it2 = tmp.begin();
		for (; it1 != container_type::end(); ++it1, ++it2)
		{
			it1->splice(it1->begin(), it2);
		}
	}

	template<typename ... Args>
	cell_type MakeCell(Args ... args) const
	{
		return std::move(cell_type(std::forward<Args...>(args)..., allocator));
	}

	/**
	 *  Traversal each cell, include boundary cells.
	 *
	 * @param fun (cell_type & cell,index_type const & s )
	 */
	template<typename Fun, typename ...Args>
	void ForAllCell(Fun const & fun, Args ...args)
	{
		mesh_->ForAll(GEOMETRY_TYPE,

		[this,&fun,&args...](index_type const & s)
		{
			fun(this->operator[](s),s,MAKE_CACHE(args,s)...);
		}

		);

	}
	template<typename Fun, typename ... Args>
	void ForAllCell(Fun const & fun, Args ... args) const
	{
		mesh_->ForAll(GEOMETRY_TYPE,

		[&](index_type const & s)
		{
			fun(this->operator[](s),s, MAKE_CACHE(args,s)...);
		}

		);

	}

	template<typename Fun, typename ...Args>
	void ForAllParticle(Fun const & fun, Args ... args)
	{

		mesh_->ForAll(GEOMETRY_TYPE,

		[&](index_type const & s)
		{
			ForParticlesInCell(s, fun, MAKE_CACHE(args,s)...);
		}

		);

	}

	template<typename Fun, typename ...Args>
	void ForAllParticle(Fun const & fun, Args ... args) const
	{
		mesh_->ForAll(GEOMETRY_TYPE,

		[&](index_type const & s)
		{
			ForParticlesInCell(s, fun, MAKE_CACHE(args,s)...);
		}

		);
	}

	template<typename Fun, typename ... Args>
	void ForParticlesInCell(index_type const &s, Fun const & fun,
			Args & ... args)
	{
		for (auto & p : this->operator[](s))
		{
			fun(p, std::forward<Args>(args)...);
		}
	}

	template<typename Fun, typename ... Args>
	void ForParticlesInCell(index_type const &s, Fun const & fun,
			Args & ... args) const
	{
		for (auto const& p : this->operator[](s))
		{
			fun(p, std::forward<Args>(args)...);
		}

	}

	const allocator_type& get_allocator() const
	{
		return allocator_;
	}

private:

	allocator_type allocator_;

}
;

template<typename TM, typename Engine>
class Particle: public ParticleBase<TM, typename Engine::Point_s>
{
public:

	typedef TM mesh_type;
	typedef Engine engine_type;
	typedef Particle<mesh_type, engine_type> this_type;
	typedef ParticleBase<TM, typename Engine::Point_s> base_type;

	template<typename ... Args>
	Particle(Args ...args) :
			base_type(std::forward<Args>(args)...)
	{
	}
	Particle(this_type const & r) :
			base_type(r)
	{
	}
	~Particle()
	{

	}

	template<typename RNDGen, typename ... Args>
	void InitLoad(size_t pic, RNDGen g, Args ...args)
	{
		//TODO need thread-safe RNDGenerator

		base_type::mesh_->ForAll(

		GEOMETRY_TYPE,

				[&](index_type const & s)
				{
					particle_type p0;

					p0.f=base_type::mesh_.GetCellVolume(s)/static_cast<Real>(pic);

					this->operator[](s).resize(pic, p0);

					Generator gen(base_type::mesh_->GetCellShape(s),MakeCache(args,s)...);

					for(auto &p:this->operator[](s))
					{
						gen(p,g);
					}
				}

				);
	}

	template<typename ... Args>
	inline void Push(Args & ... args)
	{
		base_type::ForAllParticle(engine_type::Push(),
				std::forward<Args>(args) ...);
	}
	template<typename ... Args>
	inline void ScatterJ(Args & ... args)
	{
		base_type::ForAllParticle(engine_type::ScatterJ(),
				std::forward<Args>(args) ...);
	}

	template<typename ... Args>
	inline void ScatterN(Args & ... args)
	{
		base_type::ForAllParticle(engine_type::ScatterN(),
				std::forward<Args>(args) ...);
	}

};

}
// namespace simpla

#endif /* PARTICLE_H_ */
