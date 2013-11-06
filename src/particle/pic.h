/*
 * pic.h
 *
 *  Created on: 2013年11月4日
 *      Author: salmon
 */

#ifndef PIC_H_
#define PIC_H_

#include <fetl/field_rw_cache.h>
#include <include/simpla_defs.h>
#include <cstddef>
#include <list>

namespace simpla
{

template<typename TM, typename TV>
class PIC: public TM::template Container<
		std::list<TV, FixedSmallObjectAllocator<TV> > >
{

	TM const & mesh_;
	static const int GEOMETRY_TYPE = 0;

public:

	typedef TM Mesh;

	typedef TV value_type;

	typedef TV particle_type;

	typedef typename TM::index_type index_type;

	typedef std::list<TV, FixedSmallObjectAllocator<TV> > cell_type;

	typedef typename cell_type::allocator_type allocator_type;

	typedef typename TM::template Container<cell_type> container_type;

	typedef PIC<Mesh, value_type> this_type;

	PIC(Mesh const & mesh, allocator_type allocator = allocator_type()) :
			container_type(
					std::move(
							mesh.template MakeContainer<cell_type>(
									GEOMETRY_TYPE, cell_type(allocator)))), mesh_(
					mesh)
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

		mesh_.ForAllCell(

		[&](index_type const & s)
		{
			auto & cell=this->operator[](s);
			auto pt = cell.begin();
			while (pt != cell.end())
			{
				auto p = pt;
				++pt;

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
			it1->splice(it1->begin(), *it2);
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
	void ForAllParticle(Fun const & fun, Args &... args)
	{
		/***
		 *  @BUG G++ Compiler bug (g++ <=4.8), need workaround.
		 *  Bug 41933 - [c++0x] lambdas and variadic templates don't work together
		 *   http://gcc.gnu.org/bugzilla/show_bug.cgi?id=41933

		 mesh_.ForAll(GEOMETRY_TYPE,

		 [&](index_type const & s)
		 {
		 ForParticlesInCell(s, fun, MakeCache(args,s)...);
		 }

		 );**/
		mesh_.ForAllCell(
		[&](index_type const &s, auto &... args2)
		{
			ForParticlesInCell(this->operator[](s),fun,args2...);
		},

		args...

		);
	}
	template<typename Fun, typename ...Args>
	void ForAllParticle(Fun const & fun, Args &... args) const
	{

		mesh_.ForAllCell(

		[&](index_type const &s, Args2 &... args2)
		{
			ForParticlesInCell(this->operator[](s),fun,args2...);
		},

		args...

		);
	}

private:

	template<typename Fun, typename ... Args>
	void ForParticlesInCell(cell_type & cell, Fun const & fun, Args & ... args)
	{
		for (auto & p : cell)
		{
			fun(p, std::forward<Args>(args)...);
		}
	}

	template<typename Fun, typename ... Args>
	void ForParticlesInCell(cell_type & cell, Fun const & fun,
			Args & ... args) const
	{
		for (auto & p : cell)
		{
			fun(p, std::forward<Args>(args)...);
		}
	}

};

}
// namespace simpla

#endif /* PIC_H_ */
