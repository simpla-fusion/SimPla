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
template<typename ...> class _Field;

template<typename ...T>
void update_ghosts(_Field<T...>* field)
{
//	update_ghosts(field->data(), field->domain().dataset_shape());
}

}  // namespace simpla

#endif /* FIELD_UPDATE_GHOSTS_H_ */
