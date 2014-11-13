/*
 * data_space.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_DATA_STRUCTURE_DATA_SPACE_H_
#define CORE_DATA_STRUCTURE_DATA_SPACE_H_
#include "../parallel/distributed_array.h"
namespace simpla
{
/**
 *  @brief `DataSpace` define the size and shape of dataset
 *
 *  Ref. http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
 */
class DataSpace
{

public:
	~DataSpace()
	{
	}
	DataSpace() :
			global_array_(nullptr)
	{
	}
	DataSpace(std::shared_ptr<DistributedArray> garray) :
			global_array_(garray)
	{
	}

	void swap(DataSpace &);

	static DataSpace create_simple(size_t rank, size_t const d[]);

	template<size_t RANK>
	static DataSpace create_simple(nTuple<size_t, RANK> const & d)
	{
		return std::move(create_simple(RANK, &d[0]));
	}

	size_t shape(size_t dims[], size_t offset[], size_t count, size_t block[]);

	size_t size() const;

	static constexpr size_t MAX_NUM_DIMS = 10;

	size_t start[MAX_NUM_DIMS];
	size_t stride[MAX_NUM_DIMS];
	size_t count[MAX_NUM_DIMS];
	size_t block[MAX_NUM_DIMS];

	std::shared_ptr<DistributedArray> global_array_;
	size_t ndims() const;
};

}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATA_SPACE_H_ */
