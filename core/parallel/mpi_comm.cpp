/*
 * mpi_comm.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include "mpi_comm.h"

#include <stddef.h>
#include <cstdbool>
#include <iostream>
#include <memory>
#include <string>

#include "../gtl/ntuple.h"
#include "../utilities/utilities.h"

namespace simpla
{
struct MPIComm::pimpl_s
{
	enum
	{
		m_ndims_ = 3
	};

	int m_num_process_;

	int m_process_num_;

	nTuple<int, m_ndims_> m_topology_dims_;
	nTuple<int, m_ndims_> m_topology_strides_;
	nTuple<int, m_ndims_> m_topology_coord_;

	MPI_Comm m_comm_;

	std::string init(int argc, char** argv);

	int get_neighbour(int disp_i, int disp_j = 0, int disp_k = 0) const;

	nTuple<int, m_ndims_> get_coordinate(int rank) const;

	void set_topology(int nx, int ny, int nz);

	void decompose(int ndims, size_t *count, size_t * offset) const;

	int generate_object_id();
	int m_object_id_count_;

};

std::string MPIComm::pimpl_s::init(int argc, char** argv)
{

	m_comm_ = MPI_COMM_WORLD;

	m_num_process_ = 1;
	m_process_num_ = 0;
	m_topology_dims_ = 1;
	m_object_id_count_ = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(m_comm_, &m_num_process_);
	MPI_Comm_rank(m_comm_, &m_process_num_);

	LOGGER.set_mpi_comm(m_process_num_, m_num_process_);

	set_topology(m_num_process_, 1, 1);

	parse_cmd_line(argc, argv,

	[&](std::string const & opt,std::string const & value)->int
	{

		if( opt=="number_of_threads")
		{
//			string_to_value(value,&m_num_threads_);
		}
		else if( opt=="mpi_topology")
		{
			nTuple<int, m_ndims_> d;

			string_to_value(value,&d);

			set_topology(d[0], d[1], d[2]);
		}

		return CONTINUE;

	}

	);

	VERBOSE << "MPI communicator is initialized!" << std::endl;

	return "\t--number_of_threads <NUMBER>  \t, Number of threads \n"
			"\t--mpi_topology <NX NY NZ>    \t, Set topology of mpi communicator. \n";

}

int MPIComm::pimpl_s::generate_object_id()
{
	++m_object_id_count_;
	return m_object_id_count_;
}
int MPIComm::generate_object_id()
{
	//TODO need assert (id < INT_MAX)
	return pimpl_->generate_object_id();
}
void MPIComm::pimpl_s::decompose(int ndims, size_t * p_count,
		size_t * p_offset) const
{
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> offset, count;

	offset = p_offset;
	count = p_count;
	for (int n = 0; n < m_ndims_; ++n)
	{

		p_offset[n] = offset[n]
				+ count[n] * m_topology_coord_[n] / m_topology_dims_[n];

		p_count[n] = offset[n]
				+ count[n] * (m_topology_coord_[n] + 1) / m_topology_dims_[n]
				- p_offset[n];

		if (p_count[n] <= 0)
		{
			RUNTIME_ERROR(
					"Mesh decompose fail! Dimension  is smaller than process grid. "
							"[offset= " + value_to_string(offset) + ", count="
							+ value_to_string(count) + " ,process grid="
							+ value_to_string(m_topology_coord_));
		}
	}

}
void MPIComm::pimpl_s::set_topology(int nx, int ny, int nz)
{

	if (nx * ny * nz != m_num_process_)
	{
		RUNTIME_ERROR("MPI topology is invalid!");
	}

	m_topology_dims_[0] = nx;
	m_topology_dims_[1] = ny;
	m_topology_dims_[2] = nz;

	m_topology_strides_[0] = 1;
	m_topology_strides_[1] = m_topology_dims_[0];
	m_topology_strides_[2] = m_topology_dims_[1] * m_topology_strides_[1];

	m_topology_coord_ = get_coordinate(m_process_num_);
}
nTuple<int, 3> MPIComm::pimpl_s::get_coordinate(int rank) const
{
	nTuple<int, 3> coord;
	coord[2] = rank / m_topology_strides_[2];
	coord[1] = (rank - (coord[2] * m_topology_strides_[2]))
			/ m_topology_strides_[1];
	coord[0] = rank % m_topology_dims_[0];
	return std::move(coord);
}
int MPIComm::pimpl_s::get_neighbour(int disp_i, int disp_j, int disp_k) const
{
	nTuple<int, 3> coord;

	coord[0] = (m_topology_coord_[0] + m_topology_dims_[0] + disp_i)
			% m_topology_dims_[0];
	coord[1] = (m_topology_coord_[1] + m_topology_dims_[1] + disp_j)
			% m_topology_dims_[1];
	coord[2] = (m_topology_coord_[2] + m_topology_dims_[2] + disp_k)
			% m_topology_dims_[2];

	return inner_product(coord, m_topology_strides_);
}

MPIComm::MPIComm()
		: pimpl_(nullptr)
{
}

MPIComm::MPIComm(int argc, char** argv)
		: pimpl_(nullptr)
{
	init(argc, argv);
}
MPIComm::~MPIComm()
{
	close();
}
void MPIComm::close()
{
	if (pimpl_->m_comm_ != MPI_COMM_NULL)
	{
		MPI_Finalize();

		pimpl_->m_comm_ = MPI_COMM_NULL;

		VERBOSE << "MPI Communicator is closed!" << std::endl;
	}

}
MPI_Comm MPIComm::comm()
{
	return pimpl_->m_comm_;
}
MPI_Info MPIComm::info()
{
	return MPI_INFO_NULL;
}
void MPIComm::barrier()
{
	MPI_Barrier(comm());
}

bool MPIComm::is_valid() const
{
	return (!!pimpl_) && pimpl_->m_comm_ != MPI_COMM_NULL;
}

int MPIComm::process_num() const
{
	if (!pimpl_)
	{
		return 0;
	}
	else
	{
		return pimpl_->m_process_num_;
	}
}

int MPIComm::num_of_process() const
{
	if (!pimpl_)
	{
		return 1;
	}
	else
	{
		return pimpl_->m_num_process_;
	}
}
std::string MPIComm::init(int argc, char** argv)
{
	if (!pimpl_)
	{
		pimpl_ = std::unique_ptr<pimpl_s>(new pimpl_s);
	}
	return pimpl_->init(argc, argv);
}

void MPIComm::topology(int nx, int ny, int nz)
{
	if (!!pimpl_)
	{
		pimpl_->set_topology(nx, ny, nz);
	}
}

int MPIComm::get_neighbour(int disp_i, int disp_j, int disp_k) const
{
	if (!!pimpl_)
	{
		return 0;
	}
	else
	{
		return pimpl_->get_neighbour(disp_i, disp_j, disp_k);
	}
}
nTuple<int, 3> MPIComm::topology() const
{
	return (pimpl_->m_topology_dims_);
}
nTuple<int, 3> MPIComm::get_coordinate(int rank) const
{
	return std::move(pimpl_->get_coordinate(rank));
}
void MPIComm::decompose(int ndims, size_t *count, size_t * offset) const
{
	pimpl_->decompose(ndims, count, offset);
}

//void MPIComm::barrier()
//{
//	if (comm_ != MPI_COMM_NULL)
//		MPI_Barrier(comm_);
//}
//
//void MPIComm::set_num_of_threads(int num)
//{
//	int local_num_cpu = std::thread::hardware_concurrency();
//	m_num_threads_ = std::min(num, local_num_cpu);
//}
//unsigned int MPIComm::get_num_of_threads() const
//{
//	return m_num_threads_;
//}

}// namespace simpla
