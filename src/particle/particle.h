/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <include/simpla_defs.h>
#include <cstddef>
#include <list>

namespace simpla
{

template<typename T> using PIC=std::list<T, FixedSmallObjectAllocator<T> >;
//std::map<size_t, T>;

template<typename T, typename TGeometry>
class Particle: public TGeometry, public TGeometry::Container<PIC<T> >
{
public:
	typedef typename PIC<T>::allocator_type allocator_type;

	typedef TGeometry Geometry;

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

	void push_back(T && pp)
	{
		try
		{
			T p(pp);

			container_type::at(Geometry::get_cell_num(p)).push_back_emplace(
					std::move(p));
		} catch (...)
		{

		}
	}

	void sort()
	{

		container_type tmp(
				std::move(geometry.makeContainer(pic_type(allocator))))
		);

		for (auto pt : *this)
		{
			auto it = pt.cbegin();

			while (it != pt.cend())
			{
				auto p = it;
				++it;

				auto j = Geometry::get_cell_num(*p);

				if (this->at(j) != pt)
				{

					try
					{
						tmp.at(j).slice(container_type::at(j).end(), pt, p);
					} catch (...)
					{
						pt.erase(p);
					}
				}
			}
		}
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

	void init_all(size_t num_pic, typename container_type::iterator const & b,
			typename container_type::iterator const & e)
	{
		for (auto pt = b; pt != e; ++pt)
		{
			pt->resize(num_pic);

			for (auto p : *pt)
			{
				generator_(p);
			}

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
