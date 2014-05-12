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

template<typename TM, int IFORM, typename TV> inline std::string Save(Field<TM, IFORM, TV>
const & d, std::string const & name)
{
	int rank = d.GetDataSetShape();

	size_t global_dims[rank];
	size_t local_dims[rank];
	size_t start[rank];
	size_t counts[rank];

	d.GetDataSetShape(global_dims, local_dims, start, counts, nullptr /*strides*/, nullptr/*blocks*/);

	return GLOBAL_DATA_STREAM.Write(&(*d.data()), name,rank,global_dims, local_dims, start, counts,
			nullptr /*strides*/, nullptr/*blocks*/);
}
}
 // namespace simpla

#endif /* SAVE_FIELD_H_ */
