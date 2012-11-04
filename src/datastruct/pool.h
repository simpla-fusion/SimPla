/*
 * particle_pool.h
 *
 *  Created on: 2012-3-15
 *      Author: salmon
 */

#ifndef PARTICLE_POOL_H_
#define PARTICLE_POOL_H_
#include "include/simpla_defs.h"
#include "engine/object.h"

namespace simpla
{

struct Pool: public Object
{

public:
	//TODO Define a parallel iterator

	typedef Pool ThisType;

	typedef TR1::shared_ptr<ThisType> Holder;

	Pool()
	{
		// initialize node lists
	}

	virtual ~Pool()
	{
	}

	inline void resize(size_t num)
	{
	}

// Metadata ------------------------------------------------------------

	virtual bool CheckValueType(std::type_info const & info) const
	{
		return false;
	}

	virtual inline bool CheckType(std::type_info const & info) const
	{
		return (info == typeid(Pool));
	}
	virtual inline bool Empty()
	{
		return true;
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

} // namespace simpla
#endif /* PARTICLE_POOL_H_ */
