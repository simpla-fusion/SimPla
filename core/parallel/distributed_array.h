/**
 * @file distributed_array.h
 *
 *  created on: 2014-5-30
 *      Author: salmon
 */

#ifndef DISTRIBUTED_ARRAY_H_
#define DISTRIBUTED_ARRAY_H_


#include <memory>
#include "parallel_traits.h"
#include "../dataset/dataset.h"
#include "distributed_object.h"

namespace simpla
{

/* @brief  DistributedArray is used to manage the parallel
 * communication while using the n-dimensional regular array.
 *
 * @note
 *  - DistributedArray is continue in each dimension
 *
 * inspired by :
 *  - DMDA in PETSc http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/index.html
 *
 **/

struct DistributedArray : public DataSet, public DistributedObject
{

	DistributedArray(DataType const &d_type, DataSpace const &d_space);

	DistributedArray(const DistributedArray &);

	virtual ~DistributedArray();

	virtual void swap(DistributedArray &);

	virtual void deploy();

private:

};


}//namespace simpla

#endif /* DISTRIBUTED_ARRAY_H_ */
