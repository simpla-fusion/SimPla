/*
 * particle_pool.h
 *
 *  Created on: 2012-3-15
 *      Author: salmon
 */

#ifndef PARTICLE_POOL_H_
#define PARTICLE_POOL_H_
#include "engine/object.h"
#include "engine/context.h"
#include "pic/detail/initial_random_load.h"

namespace simpla
{
namespace pic
{
template<typename TS, typename TG>
struct ParticlePool: public Object
{

public:

	typedef TG Grid;

	typedef TS Point_s;

	typedef ParticlePool<Point_s, Grid> ThisType;

	typedef TR1::shared_ptr<ThisType> Holder;

	Grid const & grid;

	ParticlePool(ThisType const & rhs) :
			Object(rhs), grid(rhs.grid), //
			num_of_elements_(rhs.num_of_elements_), //
			element_size_in_bytes_(rhs.element_size_in_bytes_), //
			element_desc_(element_desc_)
	{

	}

	ParticlePool(Grid const &pgrid, //
			size_t element_size = Point_s::get_size_in_bytes(), //
			std::string const & element_desc = Point_s::get_type_desc()) :
			grid(pgrid), //
			num_of_elements_(0), //
			element_size_in_bytes_(element_size), //
			element_desc_(element_desc)
	{
		// initialize node lists

	}

	virtual ~ParticlePool()
	{
	}

	void resize(size_t num)
	{

		size_t num_of_cells = grid.get_num_of_cell();
		size_t m = num / num_of_cells;
		num_of_elements_ = m * num_of_cells;

		Object::alloc<Point_s>(num_of_elements_, element_size_in_bytes_);

//#pragma omp parallel for
//		for (size_t s = 0; s < num_of_cells; ++s)
//		{
//			Point_s * p = (*this)[s];
//
//			p->next = (*this)[num_of_cells + m * s];
//
//			for (int i = 0; i < pic; ++i)
//			{
//				p = p->next;
//				p->next = (*this)[num_of_cells + m * s + i];
//			}
//			p->next = NULL;
//		}
	}

	// Metadata ------------------------------------------------------------

	virtual inline size_t get_element_size_in_bytes() const
	{
		return (element_size_in_bytes_);
	}
	virtual inline std::string get_element_type_desc() const
	{
		return (element_desc_);
	}

	virtual inline bool CheckValueType(std::type_info const & info) const
	{
		return (info == typeid(Point_s));
	}

	virtual inline bool CheckType(std::type_info const & info) const
	{
		return (info == typeid(ThisType));
	}

	virtual inline int get_dimensions(size_t* shape) const
	{
		if (shape != NULL)
		{
			shape[0] = num_of_elements_;
		}
		return 1;
	}

	size_t get_num_of_elements() const
	{
		return num_of_elements_;
	}

	Point_s * operator[](size_t s)
	{
		return (reinterpret_cast<Point_s *>(reinterpret_cast<char *>(Object::get_data())
				+ (s) * element_size_in_bytes_));
	}

	Point_s const * operator[](size_t s) const
	{
		return (reinterpret_cast<Point_s const*>(Object::get_data()
				+ (s) * element_size_in_bytes_));
	}

	virtual void Sort()
	{
	}

private:

	size_t num_of_elements_;
	size_t element_size_in_bytes_;
	std::string element_desc_;
};

} // namespace pic
} // namespace simpla
#endif /* PARTICLE_POOL_H_ */
