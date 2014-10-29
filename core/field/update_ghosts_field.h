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

template<typename TM, unsigned int iform, typename TC, typename ...Args>
void update_ghosts(_Field<TC, Domain<TM, iform> >* field, Args &&...args)
{
	typedef _Field<TC, Domain<TM, iform> > field_type;

	typedef typename field_traits<field_type>::value_type value_type;

//	static constexpr size_t iform = field_traits<field_type>::iform;

	auto const & global_array = field->domain().manifold().global_array_;

	value_type* data = &(*field->data());

	if (iform == VERTEX || iform == VOLUME)
	{
		update_ghosts(data, global_array, std::forward<Args >(args)...);
	}
	else
	{
		update_ghosts(reinterpret_cast<nTuple<value_type, 3>*>(data),
				global_array, std::forward<Args >(args)...);
	}
}
template<typename ...Others, typename ...Args>
void update_ghosts(_Field<Others...>* field, Args && ...args)
{

}

}  // namespace simpla

#endif /* FIELD_UPDATE_GHOSTS_H_ */
