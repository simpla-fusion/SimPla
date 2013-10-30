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

template<typename T> using PIC=std::list<T, FixedSmallObjectAllocator<T> >;

template<typename T, typename TM>
class Particle: public Geometry<TM, 0>,
		public Geometry<TM, 0>::Container<PIC<T> >
{
public:
	typedef typename PIC<T>::allocator_type allocator_type;

	typedef Geometry<TM, 0> Geometry;

	typedef T value_type;

	typedef PIC<value_type> pic_type;

	typedef typename Geometry::Container<PIC<T> > container_type;

	typedef Particle<Geometry, value_type> this_type;

	Particle(Geometry const & geometry, allocator_type allocator =
			allocator_type()) :
			Geometry(geometry), container_type(
					std::move(geometry.make_container(pic_type(allocator)))), allocator_(
					allocator)
	{
	}

	Particle(this_type const & r) = default;
	Particle(this_type && r) = default;
	~Particle() = default;

	void sort()
	{

		container_type tmp(
				std::move(Geometry::MakeContainer(pic_type(allocator))))
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

				auto j = mesh->GetCellIndex(p->X);

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
