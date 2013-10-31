/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <fetl/geometry.h>
#include <include/simpla_defs.h>
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

		mesh->ForEachAll(0,

		[&](index_type const & s)
		{
			auto & cell= this->operator[](s);

			auto pt = cell.begin();
			while (pt != cell.end())
			{
				auto p = it;
				++it;

				auto j = mesh_.SearchCell(s,p->X);

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
	 *  Traversal particles.
	 *
	 * @param fun ( paritlce_type &  )
	 */
	template<typename Fun>
	void ForEachParticle(Fun const & fun)
	{
		mesh->ForEachAll(GEOMETRY_TYPE,

		[this,&fun](index_type const & s)
		{
			auto & cell= this->operator[](s);
			for(auto & p:cell)
			{
				fun(p);
			}
		}

		);
	}

	/**
	 *  Traversal particles. (const read only version)
	 *
	 * @param fun ( paritlce_type const &  )
	 */
	template<typename Fun, typename ...Args>
	void ForEachParticle(Fun const & fun, Args ... args) const
	{
		mesh->ForEachAll(GEOMETRY_TYPE,

		[this,&fun,&args...](index_type const & s)
		{
			auto const & cell= this->operator[](s);
			for(auto const & p:cell)
			{
				fun(p,args...);
			}
		}

		);
	}

	/**
	 *  Traversal each cell, include boundary cells.
	 *
	 * @param fun (cell_type & cell,index_type const & s )
	 */
	template<typename Fun>
	void ForEachCell(Fun const & fun)
	{
		mesh->ForEachAll(GEOMETRY_TYPE,

		[this,&fun,&args...](index_type const & s)
		{
			fun(this->operator[](s),s);
		});

	}
	template<typename Fun>
	void ForEachCell(Fun const & fun) const
	{
		mesh->ForEachAll(GEOMETRY_TYPE,

		[this,&fun,&args...](index_type const & s)
		{
			fun(this->operator[](s),s);
		});

	}

	const allocator_type& get_allocator() const
	{
		return allocator_;
	}

private:

	allocator_type allocator_;

}
;

}
// namespace simpla

#endif /* PARTICLE_H_ */
