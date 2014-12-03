/*
 * dataspace.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_DATA_STRUCTURE_DATASPACE_H_
#define CORE_DATA_STRUCTURE_DATASPACE_H_

#include <stddef.h>
#include <string>
#include <tuple>

#include "../utilities/ntuple.h"

namespace simpla
{

struct DataSet;
/**
 *  @brief `DataSpace` define the size and  shape of data set in memory
 *
 *  Ref. http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
 */
class DataSpace
{

public:

	DataSpace();

	DataSpace(DataSpace const &);

	~DataSpace();

	void swap(DataSpace &);

	bool is_valid() const;

	Properties & properties(std::string const& key = "");

	Properties const& properties(std::string const& key = "") const;

	static DataSpace create_simple(size_t rank, size_t const d[]);

	template<size_t RANK>
	static DataSpace create_simple(nTuple<size_t, RANK> const & d)
	{
		return std::move(create_simple(RANK, &d[0]));
	}

	void init(size_t nd, size_t const * start, size_t const * count, size_t gw =
			2);

	bool sync_ghosts(DataSet *ds, size_t flag = 0);

	size_t num_of_dims() const;

	/**
	 * dimensions of global data
	 * @return <global start, global count>
	 */
	std::tuple<size_t const *, size_t const *> global_shape() const;

	/**
	 * dimensions of data in local memory
	 * @return <local start, local count>
	 */
	std::tuple<size_t const *, size_t const *> local_shape() const;

	/**
	 * logical shape of data in local memory, which  is the result of select_hyperslab
	 * @return <strat,strides,count,block>
	 */
	std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> shape() const;

	/**
	 *  select a hyper rectangle from local data
	 * @param start
	 * @param count
	 * @param strides
	 * @param block
	 * @return
	 */
	bool select_hyperslab(size_t const * start, size_t const * count,
			size_t const * strides = nullptr, size_t const * block = nullptr);
private:

	struct pimpl_s;
	pimpl_s* pimpl_;

};
template<typename ... Args>
DataSpace make_dataspace(Args && ... args)
{
	return DataSpace::create_simple(std::forward<Args>(args)...);
}
}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATASPACE_H_ */
