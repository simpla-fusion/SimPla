/*
 * particle_pool.h
 *
 *  Created on: 2012-3-15
 *      Author: salmon
 */

#ifndef PARTICLE_POOL_H_
#define PARTICLE_POOL_H_
#include "engine/object.h"
#include "engine/modules.h"
#include "detail/initial_random_load.h"

namespace simpla
{
namespace pic
{
template<typename TS, typename TG>
struct ParticlePool: public Object
{

public:
	//TODO Define a parallel iterator

	typedef TG Grid;

	typedef TS Point_s;

	typedef ParticlePool<Point_s, Grid> ThisType;

	typedef TR1::shared_ptr<ThisType> Holder;

	Grid const & grid;

	ParticlePool(Grid const &pgrid, size_t element_size = sizeof(Point_s),
			std::string desc = "") :
			Object(element_size, desc), grid(pgrid)

	{
		// initialize node lists

	}

	virtual ~ParticlePool()
	{
	}

	inline void resize(size_t num)
	{
		Object::ReAlloc(&num);
	}


// Metadata ------------------------------------------------------------

	virtual inline bool CheckValueType(std::type_info const & info) const
	{
		return (info == typeid(Point_s));
	}

	virtual inline bool CheckType(std::type_info const & info) const
	{
		return (info == typeid(ThisType));
	}

	Point_s & operator[](size_t s)
	{
		return (*reinterpret_cast<Point_s *>(Object::get_data(s)));
	}

	Point_s const & operator[](size_t s) const
	{
		return (*reinterpret_cast<Point_s const*>(Object::get_data(s)));
	}

	virtual void Sort()
	{
	}

	//	void resize(size_t num)
	//	{
	//
	//		size_t num_of_cells = grid.get_num_of_cell();
	//		size_t m = num / num_of_cells;
	//		num_of_elements_ = m * num_of_cells;
	//
	//		Object::alloc<Point_s>(num_of_elements_, element_size_in_bytes_);
	//
	////#pragma omp parallel for
	////		for (size_t s = 0; s < num_of_cells; ++s)
	////		{
	////			Point_s * p = (*this)[s];
	////
	////			p->next = (*this)[num_of_cells + m * s];
	////
	////			for (int i = 0; i < pic; ++i)
	////			{
	////				p = p->next;
	////				p->next = (*this)[num_of_cells + m * s + i];
	////			}
	////			p->next = NULL;
	////		}
	//	}

};

} // namespace pic
} // namespace simpla
#endif /* PARTICLE_POOL_H_ */
