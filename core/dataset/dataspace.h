/**
 * @file dataspace.h
 *
 *  Created on: 2014年11月10日
 *  @author: salmon
 */

#ifndef CORE_DATASET_DATASPACE_H_
#define CORE_DATASET_DATASPACE_H_

#include <stddef.h>
#include <map>
#include <tuple>

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
	Properties properties;

	// Creates a null dataspace
	DataSpace();

	DataSpace(int rank, size_t const * dims);

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

	static DataSpace create_simple(int rank, const size_t * dims);

	DataSpace create_distributed_space(size_t const * gw = nullptr) const;

	void select_hyperslab(size_t const *offset, size_t const * stride,
			size_t const * count, size_t const * block = nullptr);

	bool is_valid() const;

	bool is_distributed() const;

	bool is_simple() const
	{
		/// TODO support  complex selection of data space
		/// @ref http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
		return is_valid() && (!is_distributed());
	}

	/**
	 * @return <ndims,dimensions,start,count,stride,block>
	 */
	std::tuple<size_t, size_t const *, size_t const *, size_t const *,
			size_t const *, size_t const *> shape() const;

	std::tuple<size_t, size_t const *, size_t const *, size_t const *,
			size_t const *, size_t const *> global_shape() const;

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
