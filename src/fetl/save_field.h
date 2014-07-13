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
template<typename, unsigned int, typename > class Field;

template<typename TM, unsigned int IFORM, typename TV>
std::string save(std::string const & name,
        Field<TM, IFORM, DenseContainer<typename TM::compact_index_type, TV>> const & d, unsigned int flag = 0UL)
{
	typedef typename Field<TM, IFORM, DenseContainer<typename TM::compact_index_type, TV>>::value_type value_type;
	int rank = d.get_dataset_shape();
	size_t global_begin[rank];
	size_t global_end[rank];
	size_t local_outer_begin[rank];
	size_t local_outer_end[rank];
	size_t local_inner_begin[rank];
	size_t local_inner_end[rank];

	d.get_dataset_shape(

	static_cast<size_t*>(global_begin), static_cast<size_t*>(global_end),

	static_cast<size_t*>(local_outer_begin), static_cast<size_t*>(local_outer_end),

	static_cast<size_t*>(local_inner_begin), static_cast<size_t*>(local_inner_end)

	);

	return simpla::save(name, d.data().get(), rank,

	static_cast<size_t*>(global_begin), static_cast<size_t*>(global_end),

	static_cast<size_t*>(local_outer_begin), static_cast<size_t*>(local_outer_end),

	static_cast<size_t*>(local_inner_begin), static_cast<size_t*>(local_inner_end),

	(d.mesh.is_fast_first() ? DataStream::SP_FAST_FIRST : 0UL) | flag

	);

}
}
// namespace simpla

#endif /* SAVE_FIELD_H_ */
