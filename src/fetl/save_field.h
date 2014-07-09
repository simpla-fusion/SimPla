/*
 * save_field.h
 *
 *  created on: 2013-12-21
 *      Author: salmon
 */

#ifndef SAVE_FIELD_H_
#define SAVE_FIELD_H_

#include "../utilities/container_dense.h"
#include "../io/data_stream.h"

namespace simpla
{
template<typename, unsigned int , typename > class Field;

template<typename TM,  unsigned int  IFORM, typename TV>
std::string save(std::string const & name,
        Field<TM, IFORM, DenseContainer<typename TM::compact_index_type, TV>> const & d)
{
	int rank = d.get_dataset_shape();
	size_t global_begin[rank];
	size_t global_end[rank];
	size_t global_count[rank];
	size_t local_outer_begin[rank];
	size_t local_outer_end[rank];
	size_t local_outer_count[rank];
	size_t local_inner_begin[rank];
	size_t local_inner_end[rank];
	size_t local_inner_count[rank];

	d.get_dataset_shape(static_cast<size_t*>(global_begin), static_cast<size_t*>(global_end),
	        static_cast<size_t*>(local_outer_begin), static_cast<size_t*>(local_outer_end),
	        static_cast<size_t*>(local_inner_begin), static_cast<size_t*>(local_inner_end));

	for (int i = 0; i < rank; ++i)
	{
		global_count[i] = global_end[i] - global_begin[i];

		local_outer_count[i] = local_outer_end[i] - local_outer_begin[i];

		local_inner_count[i] = local_inner_end[i] - local_inner_begin[i];
	}

	return simpla::save(name, d.data().get(), rank, static_cast<size_t*>(global_begin),
	        static_cast<size_t*>(global_count), static_cast<size_t*>(local_outer_begin),
	        static_cast<size_t*>(local_outer_count), static_cast<size_t*>(local_inner_begin),
	        static_cast<size_t*>(local_inner_count));

}
}
// namespace simpla

#endif /* SAVE_FIELD_H_ */
