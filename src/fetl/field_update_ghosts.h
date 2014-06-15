/*
 * field_update_ghosts.h
 *
 *  Created on: 2014年6月15日
 *      Author: salmon
 */

#ifndef FIELD_UPDATE_GHOSTS_H_
#define FIELD_UPDATE_GHOSTS_H_

#include "../parallel/update_ghosts.h"

namespace simpla
{

template<typename TG, int IFORM, typename TV> class Field;

template<typename TM, int IFORM, typename TV, typename ...Others>
void UpdateGhosts(Field<TM, IFORM, TV>* field, Others const &...others)
{

	auto const & global_array = field->mesh.global_array_;

	TV* data = &(*field->data());

	if (IFORM == VERTEX || IFORM == VOLUME)
	{
		UpdateGhosts(data, global_array, std::forward<Others const &>(others)...);
	}
	else
	{
		UpdateGhosts(reinterpret_cast<nTuple<3, TV>*>(data), global_array, std::forward<Others const &>(others)...);
	}
}

}  // namespace simpla

#endif /* FIELD_UPDATE_GHOSTS_H_ */
