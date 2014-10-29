/*
 * save_field.h
 *
 *  created on: 2013-12-21
 *      Author: salmon
 */

#ifndef SAVE_FIELD_H_
#define SAVE_FIELD_H_

#include "../io/data_stream.h"

namespace simpla
{
template<typename ... > class _Field;

template<typename ... T>
std::string save(std::string const & url, _Field<T...> const & d,
		unsigned int flag = 0UL)
{
	if (d.empty())
	{
		return "null";
	}

//	typedef typename field_traits<_Field<T...>>::value_type value_type;
//
	int rank = d.dataset_shape();

	size_t global_begin[rank];
	size_t global_end[rank];
	size_t local_outer_begin[rank];
	size_t local_outer_end[rank];
	size_t local_inner_begin[rank];
	size_t local_inner_end[rank];

	d.dataset_shape(

	static_cast<size_t*>(global_begin),

	static_cast<size_t*>(global_end),

	static_cast<size_t*>(local_outer_begin),

	static_cast<size_t*>(local_outer_end),

	static_cast<size_t*>(local_inner_begin),

	static_cast<size_t*>(local_inner_end)

	);

	return simpla::save(url, d.data(),

	rank,

	static_cast<size_t*>(global_begin),

	static_cast<size_t*>(global_end),

	static_cast<size_t*>(local_outer_begin),

	static_cast<size_t*>(local_outer_end),

	static_cast<size_t*>(local_inner_begin),

	static_cast<size_t*>(local_inner_end),

	flag);

}
}
// namespace simpla

#endif /* SAVE_FIELD_H_ */
