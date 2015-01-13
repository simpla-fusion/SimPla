/**
 * @file data_set.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_DATA_INTERFACE_DATA_SET_H_
#define CORE_DATA_INTERFACE_DATA_SET_H_

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <type_traits>

#include "../gtl/concept_check.h"
#include "../gtl/properties.h"
#include "../gtl/type_traits.h"
#include "../utilities/memory_pool.h"
#include "data_space.h"
#include "data_type.h"

namespace simpla
{

/**
 * @ingroup data_interface
 *
 *
 * @brief Describe structure of data in the memory.
 *
 * A DataSet is composed of a pointer to raw data , a description
 * of element data type (DataType), a description of memory layout of
 * data set (DataSpace),and a container of meta data (Properties).
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

namespace _impl
{
HAS_MEMBER_FUNCTION(dataset)
}  // namespace _impl

template<typename T>
auto make_dataset(T & d)->
typename std::enable_if<_impl:: has_member_function_dataset<T,void>::value,
decltype(d.dataset())>::type
{
	return std::move(d.dataset());
}

template<typename T>
auto make_dataset(T * d)->
typename std::enable_if< _impl:: has_member_function_dataset<T,void>::value,
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
/**@}*/

}  // namespace simpla

#endif /* CORE_DATA_INTERFACE_DATA_SET_H_ */
