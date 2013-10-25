/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <engine/object.h>
#include <ext/mt_allocator.h>
#include <include/simpla_defs.h>
#include <cstddef>
#include <list>
#include <vector>

namespace simpla
{

template<typename T> using PIC=std::list<T, FixedSmallObjectAllocator<T> >;
//std::map<size_t, T>;

template<typename T, typename TGeometry>
class Particle: public TGeometry,
		public TGeometry::Container<PIC<T> >,
		public Object
{
public:
	typedef typename PIC<T>::allocator_type allocator_type;

	typedef TGeometry Geometry;

	typedef T value_type;

	typedef PIC<value_type> pic_type;

	typedef typename Geometry::Container<PIC<T> > container_type;

	typedef Particle<Geometry, value_type> this_type;

	Particle(Geometry const & geometry,

	allocator_type allocator = allocator_type()) :

			Geometry(geometry),

			container_type(geometry.make_container(pic_type(allocator))),

			allocator_(allocator)
	{
	}
	Particle(this_type const & r) :
			Geometry(r), container_type(r), allocator_(r.allocator_)

	{
	}
	~Particle()
	{
	}

	void push(T && p)
	{
		try
		{
			container_type::at(Geometry::get_cell_num(p)).push_back(Lp);
		} catch (...)
		{

		}
	}

	void sort()
	{

		container_type tmp(Geometry::get_num_of_ele(), pic_type(allocator_))

		for (size_t i = 0, max = container_type::size(); i < max; ++i)
		{
			auto it = container_type::at(i).cbegin();

			while (it != container_type::at(i).cend())
			{
				auto p = it;
				++it;

				size_t j = Geometry::get_cell_num(*p);

				if (j == i)
				{
					continue;
				}
				else
				{
					try
					{
						tmp.at(j).slice(container_type::at(j).end(),
								container_type::at(i), p);
					} catch (...)
					{
						container_type::at(i).erase(p);
					}
				}
			}
		}
		auto it1 = container_type::begin();
		auto it2 = tmp.begin();
		for (; it1 != container_type::end(); ++it1, ++it2)
		{
			it1->slice(it1->begin(), it2);
		}
	}

	const allocator_type& getAllocator() const
	{
		return allocator_;
	}

private:

	allocator_type allocator_;

//	struct iterator
//	{
//		typedef typename std::vector<Tree>::iterator t_iterator;
//		t_iterator t_it;
//		t_iterator t_end;
//		typedef typename Tree::iterator l_iterator;
//		l_iterator l_it;
//
//		iterator()
//		{
//
//		}
//		inline iterator operator ++()
//		{
//			++l_it;
//			if (l_it == t_it.end())
//			{
//				++t_it;
//				if (t_it != t_end)
//				{
//					l_it = t_it.front();
//				}
//			}
//			return *this;
//		}
//		inline T & operator*()
//		{
//			return *l_it;
//		}
//		inline T const& operator*() const
//		{
//			return *l_it;
//		}
//
//	};

}
;

}
// namespace simpla

#endif /* PARTICLE_H_ */
