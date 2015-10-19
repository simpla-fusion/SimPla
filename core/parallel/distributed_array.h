/**
 * @file distributed_array.h
 *
 *  created on: 2014-5-30
 *      Author: salmon
 */

#ifndef DISTRIBUTED_ARRAY_H_
#define DISTRIBUTED_ARRAY_H_


#include <memory>

#include "../dataset/dataset.h"

#include "parallel_traits.h"

#include "distributed.h"

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

template<> void Distributed<DataSet>::deploy();

}//namespace simpla

#endif /* DISTRIBUTED_ARRAY_H_ */
