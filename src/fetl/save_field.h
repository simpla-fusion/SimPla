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

template<typename TG, int IFORM, typename TV> inline std::string Dump(Field<TG, IFORM, TV>
const & d, std::string const & name, bool flag)
{
	return DataDumper<TV>(d.data().get(), name, d.GetShape(), flag).GetName();
}
}  // namespace simpla

#endif /* SAVE_FIELD_H_ */
