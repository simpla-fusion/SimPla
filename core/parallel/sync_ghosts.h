/*
 * @file sync_ghosts.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_SYNC_GHOSTS_H_
#define CORE_PARALLEL_SYNC_GHOSTS_H_

namespace simpla
{

template<typename T>
void sync_ghosts(T * f, size_t flag = 0)
{
	sync_ghosts(f->dataset(), flag);
}
class DataSet;

int sync_ghosts(DataSet * data, size_t flag = 0);

}  // namespace simpla

#endif /* CORE_PARALLEL_SYNC_GHOSTS_H_ */
