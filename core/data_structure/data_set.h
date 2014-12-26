/*
 * data_set.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_DATA_STRUCTURE_DATA_SET_H_
#define CORE_DATA_STRUCTURE_DATA_SET_H_

#include <memory>

#include "../utilities/utilities.h"
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

HAS_MEMBER_FUNCTION(dataset)

template<typename T>
auto make_dataset(T & d)->
typename std::enable_if< has_member_function_dataset<T,void>::value,
decltype(d.dataset())>::type
{
	return std::move(d.dataset());
}

template<typename T>
auto make_dataset(T * d)->
typename std::enable_if< has_member_function_dataset<T,void>::value,
decltype(d->dataset())>::type
{
	return std::move(d->dataset());
}

template<typename T>
DataSet make_dataset(int rank, size_t const * dims, Properties const & prop =
		Properties())
{
	DataSet res;
	res.datatype = make_datatype<T>("");
	res.dataspace.init(rank, dims);
	res.data = sp_make_shared_array<T>(res.dataspace.size());
	res.attribute = prop;
	return std::move(res);
}

template<typename T>
DataSet make_dataset(T * p, int rank, size_t const * dims,
		Properties const & prop = Properties())
{

	DataSet res;

	res.datatype = make_datatype<T>();
	res.dataspace.init(rank, dims);
	res.data = std::shared_ptr<void>(
			const_cast<void*>(reinterpret_cast<typename std::conditional<
					std::is_const<T>::value, void const *, void *>::type>(p)),
			do_nothing());
	res.attribute = prop;

	return std::move(res);
}

template<typename T>
DataSet make_dataset(std::shared_ptr<T> p, int rank, size_t const * dims,
		Properties const & prop = Properties())
{

	DataSet res;
	res.data = p;
	res.datatype = make_datatype<T>();
	res.dataspace = make_dataspace(rank, dims);
	res.attribute = prop;

	return std::move(res);
}
}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_DATA_SET_H_ */
