/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_
<<<<<<< HEAD

#include "engine/object.h"
#include <list>
#include <vector>
//need  libstdc++
#include <ext/mt_allocator.h>

namespace simpla
{
//struct SampleNode
//{
//	typename TGeometry::CoordinateType Z;
//
//	TSample f;
//};

template<typename T, typename TGeometry>
class Particle: public TGeometry, public std::vector<
		std::list<T, typename __gnu_cxx::__mt_alloc<T> > >, public Object
{
public:
	//TODO: default allocator is not enough, we need pool_allocator or __mt_alloc

	typedef typename __gnu_cxx::__mt_alloc<T> allocator_type;

	typedef std::list<T, allocator_type> ParticlesList;

	typedef std::vector<ParticlesList> StorageType;

	typedef TGeometry Geometry;

	typedef T ValueType;

	typedef Particle<Geometry, ValueType> ThisType;

	Particle() :
			BaseType(Geometry::get_num_of_ele(), ParticlesList(allocator_))
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
		BaseType tmp(Geometry::get_num_of_ele(), ParticlesList(allocator_))

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
=======
#include "include/simpla_defs.h"
#include "fetl/fetl.h"
#include "engine/object.h"
namespace simpla
{
template<typename TG>
struct Particle:public CompoundObject
{

	DEFINE_FIELDS(TG)
	ZeroForm n;
	VecZeroForm J;

};
>>>>>>> ddb1baf4864f73bec4047c704d79f5c9a1152544

}
// namespace simpla

#endif /* PARTICLE_H_ */
