/*
 * field_update_ghosts.h
 *
 *  created on: 2014-6-15
 *      Author: salmon
 */

#ifndef FIELD_UPDATE_GHOSTS_H_
#define FIELD_UPDATE_GHOSTS_H_

#include "../parallel/distributed_array.h"
namespace simpla
{
template<typename, size_t> class Domain;
template<typename ...> class _Field;

template<typename TM, unsigned int IFORM, typename TC, typename ...Others,
		typename ...Args>
void update_ghosts(_Field<TC, Domain<TM, IFORM>, Others...>* field,
		Args &&...args)
{

	UNIMPLEMENT;
}
template<typename ...Others>
void update_ghosts(_Field<Others...>* fields)
{

//	typedef TV value_type;
//	auto const & global_array = field->mesh.global_array_;
//
//	value_type* data = &(*field->data());
//
//	if (IFORM == VERTEX || IFORM == VOLUME)
//	{
//		update_ghosts(data, global_array, std::forward<Others >(others)...);
//	}
//	else
//	{
//		update_ghosts(reinterpret_cast<nTuple<3, value_type>*>(data),
//				global_array, std::forward<Others >(others)...);
//	}
}

}  // namespace simpla

#endif /* FIELD_UPDATE_GHOSTS_H_ */
