/*
 * sp_object.cpp
 *
 *  Created on: 2015年3月6日
 *      Author: salmon
 */
#include "sp_object.h"
#include "../parallel/mpi_comm.h"

namespace simpla
{
//! Default constructor
SpObject::SpObject()
{
}
//! destroy.
SpObject::~SpObject()
{
}

SpObject::SpObject(const SpObject&)
{
}

bool SpObject::is_ready() const
{
	//FIXME this is not multi-thread safe

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

void SpObject::wait_to_ready()
{
	//FIXME this is not multi-thread safe
	if (!is_valid())
	{
		deploy();
	}
	if (m_mpi_requests_.size() > 0)
	{
		MPI_ERROR(MPI_Waitall( m_mpi_requests_.size(), //
				const_cast<MPI_Request*>(&m_mpi_requests_[0]),//
				MPI_STATUSES_IGNORE));
		m_mpi_requests_.clear();
	}
}

std::ostream &SpObject::print(std::ostream & os) const
{
	return properties.print(os);
}
}  // namespace simpla
