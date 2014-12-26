/*
 * dataspace.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_DATA_STRUCTURE_DATA_SPACE_H_
#define CORE_DATA_STRUCTURE_DATA_SPACE_H_

#include <stddef.h>
#include <string>
#include <tuple>
#include <memory>
#include "../utilities/utilities.h"

namespace simpla
{

struct DataSet;
/**
 *  @brief `DataSpace` define the size and  shape of data set in memory/file
 *
 *  Ref. http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
 */
class DataSpace
{
public:
	Properties properties;

	// Creates a null dataspace
	DataSpace();

	DataSpace(int rank, size_t const * dims, const size_t * gw = nullptr);

	// Copy constructor: makes a copy of the original DataSpace object.
	DataSpace(const DataSpace& other);

	// Assignment operator
	DataSpace& operator=(const DataSpace& rhs);

	// Destructor: properly terminates access to this dataspace.
	~DataSpace();

	void swap(DataSpace &);

	void init(int rank, const size_t * dims, const size_t * gw = nullptr);

	bool is_valid() const;

	bool is_simple() const
	{
		/// TODO support  complex selection of data space
		/// ref http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html

		return true;
	}

	size_t size() const;

	/**
	 * @return <ndims,dimensions,start,count,stride,block>
	 */
	std::tuple<size_t, size_t const *, size_t const *, size_t const *,
			size_t const *, size_t const *> shape() const;

	bool select_hyperslab(size_t const *offset, size_t const * count,
			size_t const * stride = nullptr, size_t const * block = nullptr);

	void decompose(int num_procs = 0, size_t const * gw = nullptr);

	void compose(size_t flag = 0UL);

	bool is_distributed() const;

	DataSpace const & local_space() const;

	std::map<int, DataSpace> const& neighgours() const
	{
		return neighgours_;
	}

private:
	struct pimpl_s;
	pimpl_s * pimpl_;

	std::map<int, DataSpace> neighgours_;

};
template<typename ... Args>
DataSpace make_dataspace(Args && ... args)
{
	return DataSpace(std::forward<Args>(args)...);
}

}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATA_SPACE_H_ */
