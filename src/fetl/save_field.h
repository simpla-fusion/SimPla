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
template<typename, typename > struct Field;

template<typename TG, typename TV> inline DataSet<TV> Data(Field<TG, TV> const & d, std::string const & name, bool flag)
{
	return std::move(DataSet<TV>(d.data(), name, d.GetShape(), flag));
}
}  // namespace simpla

#endif /* SAVE_FIELD_H_ */
