/*
 * sp_object.cpp
 *
 *  Created on: 2015年3月6日
 *      Author: salmon
 */
#include "sp_object.h"
#include "../parallel/mpi_comm.h"
#include "../parallel/mpi_aux_functions.h"
#include "../parallel/mpi_update.h"
namespace simpla
{
//! Default constructor
SpObject::SpObject()
{
	m_object_id_ =
			SingletonHolder<simpla::MPIComm>::instance().generate_object_id();
}

SpObject::SpObject(const SpObject&)
{
	m_object_id_ =
			SingletonHolder<simpla::MPIComm>::instance().generate_object_id();
}
//! destroy.
SpObject::~SpObject()
{
}

bool SpObject::is_ready() const
{
	//FIXME this is not multi-threads safe

	if (is_valid() && m_mpi_requests_.size() > 0)
	{
		int flag = 0;
		MPI_ERROR(MPI_Testall(m_mpi_requests_.size(), //
				const_cast<MPI_Request*>(&m_mpi_requests_[0]),//
				&flag, MPI_STATUSES_IGNORE));

		return flag != 0;
	}

	return false;

}
void SpObject::prepare_sync(std::vector<mpi_ghosts_shape_s> const & ghost_shape)
{
	int ndims = 3;

	auto ds = dataset();

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> l_dims;

	auto d_shape = ds.dataspace.local_shape();

	ndims = d_shape.ndims;

	l_dims = d_shape.dimensions;

	make_send_recv_list(object_id(), ds.datatype, ndims, &l_dims[0],
			ghost_shape, &m_send_recv_list_);
}
void SpObject::sync()
{
	if (m_send_recv_list_.size() > 0)
	{
		sync_update_continue(m_send_recv_list_, dataset().data.get()
#ifndef DISABLE_ASYNC_UPDATE
				, &(m_mpi_requests_)
#endif
				);
	}
}

void SpObject::wait()
{
	//FIXME this is not multi-thread safe
	if (!is_valid())
	{
		deploy();
	}

	wait_all_request(&m_mpi_requests_);
}

std::ostream &SpObject::print(std::ostream & os) const
{
	return properties.print(os);
}
}  // namespace simpla
