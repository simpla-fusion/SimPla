/*
 * dataspace.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_DATA_STRUCTURE_DATASPACE_H_
#define CORE_DATA_STRUCTURE_DATASPACE_H_
#include "../parallel/distributed_array.h"
namespace simpla
{
/**
 *  @brief `DataSpace` define the size and  shape of data set in memory
 *
 *  Ref. http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
 */
class DataSpace
{

public:
	static constexpr size_t MAX_NUM_DIMS = DistributedArray::MAX_NUM_DIMS;

	DataSpace() :
			darray_(nullptr)
	{
	}

	template<typename ...Args>
	DataSpace(Args && ... args)
	{
	}

	DataSpace(std::shared_ptr<DistributedArray> d) :
			darray_(d)
	{
		update();
	}

	~DataSpace()
	{
	}
	void swap(DataSpace &);

	static DataSpace create_simple(size_t rank, size_t const d[])
	{
		return DataSpace();
	}

	template<size_t RANK>
	static DataSpace create_simple(nTuple<size_t, RANK> const & d)
	{
		return std::move(create_simple(RANK, &d[0]));
	}

	void select(size_t ndims, size_t const* offset_, size_t const*count_);

	inline size_t get_shape(size_t *global_begin = nullptr,

	size_t *global_end = nullptr,

	size_t *local_outer_begin = nullptr,

	size_t *local_outer_end = nullptr,

	size_t *local_inner_begin = nullptr,

	size_t *local_inner_end = nullptr) const
	{
		return 0;
	}

private:

	size_t ndims = 0;

	size_t begin[MAX_NUM_DIMS];
	size_t end[MAX_NUM_DIMS];

	std::shared_ptr<DistributedArray> darray_;

	void update()
	{
	}

};
template<typename ... Args>
DataSpace make_dataspace(Args && ... args)
{
	return DataSpace::create_simple(std::forward<Args>(args)...);
}
}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATASPACE_H_ */
