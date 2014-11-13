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
 *  @brief `DataSpace` define the size and shape of dataset
 *
 *  Ref. http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
 */
class DataSpace
{

public:

	DataSpace(std::shared_ptr<DistributedArray> d,
			size_t const *block = nullptr) :
			darray_(d)
	{
		global_begin_ = 0;
		global_end_ = 1;
		local_outer_begin_ = 0;
		local_outer_end_ = 1;
		local_inner_begin_ = 0;
		local_inner_end_ = 1;
		block_ = 1;

	}

	~DataSpace()
	{
	}
	void swap(DataSpace &);

	static DataSpace create_simple(size_t rank, size_t const d[]);

	template<size_t RANK>
	static DataSpace create_simple(nTuple<size_t, RANK> const & d)
	{
		return std::move(create_simple(RANK, &d[0]));
	}

private:

	static constexpr size_t MAX_NUM_DIMS = 10;

	size_t ndims_ = 1;

	nTuple<size_t, MAX_NUM_DIMS> global_begin_;

	nTuple<size_t, MAX_NUM_DIMS> global_end_;

	nTuple<size_t, MAX_NUM_DIMS> local_outer_begin_;

	nTuple<size_t, MAX_NUM_DIMS> local_outer_end_;

	nTuple<size_t, MAX_NUM_DIMS> local_inner_begin_;

	nTuple<size_t, MAX_NUM_DIMS> local_inner_end_;

	nTuple<size_t, MAX_NUM_DIMS> block_;

	std::shared_ptr<DistributedArray> darray_;
};

}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATASPACE_H_ */
