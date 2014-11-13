/*
 * update_ghosts.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_UPDATE_GHOSTS_H_
#define CORE_PARALLEL_UPDATE_GHOSTS_H_

#include <tuple>

#include "../utilities/sp_type_traits.h"

namespace simpla
{
struct DataSpace;
struct DataType;

template<typename T>
int update_ghosts(T * f)
{
	return 0;
}

template<typename T>
auto update_ghosts(T * f)
DECL_RET_TYPE((update_ghosts(f->dataset())))

template<typename T>
auto update_ghosts(T & f)
DECL_RET_TYPE((update_ghosts(f.dataset())))

template<typename ... T>
int update_ghosts(std::tuple<T...> & dset)
{
	return update_ghosts(&*std::get<0>(dset), std::get<1>(dset),
			std::get<2>(dset));
}

int update_ghosts(void * data, DataType const & datatype,
		DataSpace const & space);

}  // namespace simpla

#endif /* CORE_PARALLEL_UPDATE_GHOSTS_H_ */
