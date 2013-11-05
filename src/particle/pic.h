/*
 * pic.h
 *
 *  Created on: 2013年11月4日
 *      Author: salmon
 */

#ifndef PIC_H_
#define PIC_H_

#include <include/simpla_defs.h>
#include <cstddef>
#include <list>

namespace simpla
{

template<typename TM, typename TV>
class PIC: public TM::Container<std::list<TV, FixedSmallObjectAllocator<TV> > >
{

	TM const & mesh_;
	static const int GEOMETRY_TYPE = 0;

public:

	typedef TM Mesh;

	typedef typename TV value_type;

	typedef typename TV particle_type;

	typedef std::list<TV, FixedSmallObjectAllocator<TV> > cell_type;

	typedef typename cell_type::allocator_type allocator_type;

	typedef typename TM::Container<cell_type> container_type;

	typedef PIC<Mesh, value_type> this_type;

	PIC(Mesh const & mesh, allocator_type allocator = allocator_type()) :
			container_type(
					std::move(
							mesh.MakeContianer(GEOMETRY_TYPE,
									cell_type(allocator)))), mesh_(mesh)
	{
	}

	PIC(this_type const & r) :
			container_type(r), mesh_(r.mesh_)
	{
	}

	PIC(this_type && r) :
			container_type(std::move(r)), mesh_(r.mesh_)
	{
	}

	~PIC() = default;

	void Sort()
	{

		container_type tmp(
				std::move(
						mesh_.MakeContainer(GEOMETRY_TYPE,
								cell_type(
										container_type::begin()->get_allocator()))));

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
		}

		);
		auto it1 = container_type::begin();
		auto it2 = tmp.begin();
		for (; it1 != container_type::end(); ++it1, ++it2)
		{
			it1->splice(it1->begin(), it2);
		}
	}

	inline void ResizeCells(size_t num_pic,
			particle_type const & default_value = particle_type())
	{
		for (auto & cell : *this)
		{
			cell.resize(num_pic, default_value);
		}

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

private:
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

};

}
// namespace simpla

#endif /* PIC_H_ */
