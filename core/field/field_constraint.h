/*
 * @file field_constraint.h
 *
 *  Created on: 2014年12月2日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_CONSTRAINT_H_
#define CORE_FIELD_FIELD_CONSTRAINT_H_
#include "../model/select.h"
#include "../utilities/log.h"
#include "field_function.h"
#include <set>
namespace simpla
{

template<typename TM, typename TV, typename TDict>
auto make_constraint(TM const & mesh, TDict const & dict)
{
	auto op = dict["Value"];

	typedef typename TM::template field_value_type<TV> field_value_type;

	return make_field_function<TM, TV>(
			select_ids_by_configure(mesh, dict["Domain"]), dict["Value"]);

}

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_CONSTRAINT_H_ */
