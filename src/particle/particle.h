/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "include/simpla_defs.h"
#include "engine/object.h"
#include <list>
#include <vector>
namespace simpla
{

template<typename T> using PIC=std::list<T, SmallObjectAllocator<T> >;

template<typename T, typename TGeometry>
class Particle: public TGeometry,
		public std::vector<PIC<T> >,
		public Object
{
public:
	typedef SmallObjectAllocator<T> allocator_type;

	typedef TGeometry Geometry;

	typedef T ValueType;

	typedef PIC<ValueType> pic_type;

	typedef std::vector<PICListType> StorageType;

	typedef Particle<Geometry, ValueType> ThisType;

	Particle() :
			BaseType(Geometry::get_num_of_ele(), pic_type(allocator_))
	{
	}
	Particle(ThisType const & r) :
			BaseType(r), allocator_(r.allocator_)
	{
	}
	~Particle()
	{
	}

	allocator_type get_allocator() const
	{
		return allocator_;
	}

	void push(T && p)
	{
		try
		{
			StorageType::at(Geometry::get_cell_num(p)).push_back(p);
		} catch (...)
		{

		}
	}

	void sort()
	{
		/**
		 * TODO need review;
		 *   some particles are sorted twice;
		 */
		BaseType tmp(Geometry::get_num_of_ele(), pic_type(allocator_))

		for (size_t i = 0, max = StorageType::size(); i < max; ++i)
		{
			auto it = StorageType::at(i).cbegin();

			while (it != StorageType::at(i).cend())
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
						tmp.at(j).slice(StorageType::at(j).end(),
								StorageType::at(i), p);
					} catch (...)
					{
						StorageType::at(i).erase(p);
					}
				}
			}
		}

		for (auto it1 = StorageType::begin(), it2 = tmp.begin();
				it1 != StorageType::end(); ++it1, ++it2)
		{
			it1->slice(it1->begin(), it2);
		}
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
