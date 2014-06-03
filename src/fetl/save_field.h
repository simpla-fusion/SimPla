/*
 * save_field.h
 *
 *  Created on: 2013年12月21日
 *      Author: salmon
 */

#ifndef SAVE_FIELD_H_
#define SAVE_FIELD_H_

#include "../io/data_stream.h"

namespace simpla
{
template<typename, int, typename > struct Field;

template<typename TM, int IFORM, typename TV>
std::string Save(std::string const & name, Field<TM, IFORM, TV> const & d)
{
	int rank = d.GetDataSetShape();
	size_t global_start[rank];
	size_t global_count[rank];
	size_t local_outer_start[rank];
	size_t local_outer_count[rank];
	size_t local_inner_start[rank];
	size_t local_inner_count[rank];

	d.GetDataSetShape(global_start, global_count, local_outer_start, local_outer_count, local_inner_start,
	        local_inner_count);

	return simpla::Save(name,

	d.data().get(),

	rank, global_start, global_count, local_outer_start, local_outer_count, local_inner_start, local_inner_count);

}
}
// namespace simpla

#endif /* SAVE_FIELD_H_ */
