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

#include "../utilities/ntuple.h"

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

	// Creates a null dataspace
	DataSpace();

	// Creates a simple dataspace
	DataSpace(int rank, const size_t * dims);

	// Copy constructor: makes a copy of the original DataSpace object.
	DataSpace(const DataSpace& other);

	// Assignment operator
	DataSpace& operator=(const DataSpace& rhs);

	// Destructor: properly terminates access to this dataspace.
	~DataSpace();

	void swap(DataSpace &);

	void init(int rank, const size_t * dims);

	bool is_valid() const;

	bool is_distributed() const;

	bool is_simple() const
	{
		/// TODO support  complex selection of data space
		/// ref http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html

		return true;
	}

	DataSpace const & global_space() const;

	/**
	 * @return <ndims,dimensions,start,count,stride,block>
	 */

	std::tuple<size_t, size_t const *, size_t const *, size_t const *,
			size_t const *, size_t const *> shape() const;

	bool select_hyperslab(size_t const *start, size_t const * count,
			size_t const * stride = nullptr, size_t const * block = nullptr);

	std::map<int, std::shared_ptr<DataSpace>> neighgours;

private:
	struct pimpl_s;
	pimpl_s *pimpl_;

};
template<typename ... Args>
DataSpace make_dataspace(Args && ... args)
{
	return DataSpace(std::forward<Args>(args)...);
}

}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATA_SPACE_H_ */
