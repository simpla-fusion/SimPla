/**
 * @file dataspace.h
 *
 *  Created on: 2014年11月10日
 *  @author: salmon
 */

#ifndef CORE_DATASET_DATASPACE_H_
#define CORE_DATASET_DATASPACE_H_

#include <stddef.h>
#include <memory>
#include <tuple>
#include <vector>

#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"
#include "../gtl/properties.h"

namespace simpla
{

struct DataSet;
/**
 * @ingroup data_interface
 * @brief  Define the size and  shape of data set in memory/file
 *  Ref. http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
 */
class DataSpace
{
public:

	// Creates a null dataspace
	DataSpace();

	DataSpace(int rank, const size_t * dims, const size_t * count = nullptr,
			const size_t * offset = nullptr, const size_t * stride = nullptr,
			const size_t * block = nullptr);

	// Copy constructor: makes a copy of the original DataSpace object.
	DataSpace(const DataSpace& other);

	// Destructor: properly terminates access to this dataspace.
	~DataSpace();

	void swap(DataSpace &);

	// Assignment operator
	DataSpace& operator=(const DataSpace& rhs)
	{
		DataSpace(rhs).swap(*this);
		return *this;
	}

	DataSpace & add_ghosts(size_t const * gw = nullptr);

	DataSpace & select_hyperslab(size_t const *offset, size_t const * stride,
			size_t const * count, size_t const * block = nullptr);

	bool is_valid() const;

	bool is_distributed() const;

	bool is_simple() const
	{
		/// TODO support  complex selection of data space
		/// @ref http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
		return is_valid() && (!is_distributed());
	}

	typedef nTuple<size_t, MAX_NDIMS_OF_ARRAY> index_tuple;

	struct shape_s
	{
		size_t ndims = 3;
		index_tuple dimensions;
		index_tuple count;
		index_tuple offset;
		index_tuple stride;
		index_tuple block;

		operator std::tuple<size_t, size_t const *, size_t const *, size_t const *,
		size_t const *, size_t const *>()
		{
			return std::make_tuple(ndims, &dimensions[0], &count[0], &offset[0],
					&stride[0], &block[0]);
		}
	};

	shape_s memory_shape() const;

	shape_s file_shape() const;

private:
	struct pimpl_s;
	std::unique_ptr<pimpl_s> pimpl_;

};
/**
 * @ingroup data_interface
 * create dataspace
 * @param args
 * @return
 */
template<typename ... Args>
DataSpace make_dataspace(Args && ... args)
{
	return DataSpace(std::forward<Args>(args)...);
}

/**@}  */

}  // namespace simpla

#endif /* CORE_DATASET_DATASPACE_H_ */
