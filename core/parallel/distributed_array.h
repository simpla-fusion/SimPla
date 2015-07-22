/**
 * @file distributed_array.h
 *
 *  created on: 2014-5-30
 *      Author: salmon
 */

#ifndef DISTRIBUTED_ARRAY_H_
#define DISTRIBUTED_ARRAY_H_

#include <stddef.h>
#include <string>
#include <tuple>
#include <memory>
namespace simpla
{
struct DataSet;

/* @brief  DistributedArray is used to manage the parallel
 * communication while using the n-dimensional regular array.
 * @note
 *  - DistributedArray is continue in each dimension, which means
 *    no `strides` or `blocks`
 *  -
 *
 * inspired by :
 *  - DMDA in PETSc
 *     http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/index.html
 *
 *
 *
 **/

struct DistributedArray
{

	DistributedArray();

	~DistributedArray();

	void decompose(size_t nd = 0, int const * dims = nullptr);

	bool sync_ghosts(DataSet* ds, size_t flag = 0) const;

private:
	struct pimpl_s;
	std::unique_ptr<pimpl_s> pimpl_;

}
;

//void update_ghosts(void * data, DataType const & data_type,
//		DistributedArray const & global_array);
//
//template<typename TV>
//void update_ghosts(std::shared_ptr<TV> data,
//		DistributedArray const & global_array)
//{
//	update_ghosts(data.get(), DataType::create<TV>(), global_array);
//}
//
//template<typename TV>
//void update_ghosts(TV * data, DistributedArray const & global_array)
//{
//	update_ghosts(data, DataType::create<TV>(), global_array);
//}
}
// namespace simpla

#endif /* DISTRIBUTED_ARRAY_H_ */
