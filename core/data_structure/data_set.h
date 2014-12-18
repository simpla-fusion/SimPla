/*
 * data_set.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_DATA_STRUCTURE_DATA_SET_H_
#define CORE_DATA_STRUCTURE_DATA_SET_H_

#include <memory>

#include "../utilities/properties.h"
#include "../utilities/memory_pool.h"
#include "data_type.h"
#include "data_space.h"

namespace simpla
{

/**
 *  @brief DataSet
 *
 *
 */

struct DataSet
{
	std::shared_ptr<void> data;
	DataType datatype;
	DataSpace dataspace;
	Properties attribute;

	bool is_valid() const
	{
		return data != nullptr && datatype.is_valid() && dataspace.is_valid();
	}
};

template<typename T>
DataSet make_dataset(int rank, size_t const * dims)
{
	DataSet res;
	res.datatype = DataType::create<T>();
	res.dataspace.init(rank, dims);
	res.data = sp_make_shared_array<T>(res.dataspace.size());
	return std::move(res);
}

}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATA_SET_H_ */
