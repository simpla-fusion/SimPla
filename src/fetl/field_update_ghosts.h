/*
 * field_update_ghosts.h
 *
 *  created on: 2014-6-15
 *      Author: salmon
 */

#ifndef FIELD_UPDATE_GHOSTS_H_
#define FIELD_UPDATE_GHOSTS_H_

#include "../utilities/container_dense.h"
#include "../parallel/distributed_array.h"
namespace simpla
{
template<typename, unsigned int> class Domain;
template<typename ...> class Field;

template<typename TM, unsigned int IFORM, typename TC, typename ...Others>
void update_ghosts(Field<Domain<TM, IFORM>, TC>* field, Others &&...others)
{

	UNIMPLEMENT;
}
template<typename TM, unsigned int IFORM, typename TV, typename ...Others>
void update_ghosts(
		Field<Domain<TM, IFORM>,
				DenseContainer<typename TM::compact_index_type, TV>>* field,
		Others &&...others)
{

	typedef TV value_type;
	auto const & global_array = field->mesh.global_array_;

	value_type* data = &(*field->data());

	if (IFORM == VERTEX || IFORM == VOLUME)
	{
		update_ghosts(data, global_array, std::forward<Others >(others)...);
	}
	else
	{
		update_ghosts(reinterpret_cast<nTuple<3, value_type>*>(data),
				global_array, std::forward<Others >(others)...);
	}
}

}  // namespace simpla

#endif /* FIELD_UPDATE_GHOSTS_H_ */
