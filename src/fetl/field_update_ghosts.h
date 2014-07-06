/*
 * field_update_ghosts.h
 *
 *  Created on: 2014-6-15
 *      Author: salmon
 */

#ifndef FIELD_UPDATE_GHOSTS_H_
#define FIELD_UPDATE_GHOSTS_H_

#include "../utilities/container_dense.h"
#include "../parallel/update_ghosts.h"

namespace simpla
{

template<typename TG, int IFORM, typename TV> class Field;
template<typename TM, int IFORM, typename TC, typename ...Others>
void UpdateGhosts(Field<TM, IFORM, TC>* field, Others &&...others)
{

	UNIMPLEMENT;
}
template<typename TM, int IFORM, typename TV, typename ...Others>
void UpdateGhosts(Field<TM, IFORM, DenseContainer<typename TM::compact_index_type, TV>>* field, Others &&...others)
{

	typedef TV value_type;
	auto const & global_array = field->mesh.global_array_;

	value_type* data = &(*field->data());

	if (IFORM == VERTEX || IFORM == VOLUME)
	{
		UpdateGhosts(data, global_array, std::forward<Others >(others)...);
	}
	else
	{
		UpdateGhosts(reinterpret_cast<nTuple<3, value_type>*>(data), global_array, std::forward<Others >(others)...);
	}
}


}  // namespace simpla

#endif /* FIELD_UPDATE_GHOSTS_H_ */
