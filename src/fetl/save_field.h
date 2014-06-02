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

template<typename TM, int IFORM, typename TV> inline std::string Save(std::string const & name, Field<TM, IFORM, TV>
const & d)
{
	int rank = d.GetDataSetShape();

	size_t global_dims[rank];
	size_t local_outer_start[rank];
	size_t local_outer_count[rank];
	size_t local_inner_start[rank];
	size_t local_inner_count[rank];

	d.GetDataSetShape(global_dims, local_outer_start, local_outer_count, local_inner_start, local_inner_count);

	return GLOBAL_DATA_STREAM.Write(name, &(*d.data()), rank,global_dims,local_outer_start,
			local_outer_count, local_inner_start, local_inner_count );

}
}
// namespace simpla

#endif /* SAVE_FIELD_H_ */
