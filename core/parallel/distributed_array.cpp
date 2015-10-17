/*
 * distributed_array.cpp
 *
 *  Created on: 2014-11-13
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_
#define CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_

#include "../gtl/utilities/log.h"

#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_update.h"

#include "distributed_array.h"

namespace simpla
{


//! Default constructor
DistributedArray::DistributedArray(DataType const &d_type, DataSpace const &d_space)
{


	DataSet::datatype = d_type;

	DataSet::dataspace = d_space;

	DataSet::data = nullptr;

}

DistributedArray::DistributedArray(DistributedArray const &other) :
		DataSet(other)
{
}

DistributedArray::~DistributedArray()
{
}

void DistributedArray::swap(DistributedArray &other)
{
	DataSet::swap(other);
}

void DistributedArray::deploy()
{
	DataSet::deploy();

	DistributedObject::add_link(*this);
}




}// namespace simpla


#endif /* CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_ */
