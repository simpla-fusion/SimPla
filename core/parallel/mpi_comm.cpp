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
#include "../utilities/log.h"
#include "../utilities/misc_utilities.h"
#include "../utilities/parse_command_line.h"

namespace simpla
{
struct MPIComm::pimpl_s
{
	enum
	{
		NUM_OF_DIMS = 3
	};

	int m_num_process_;

	int m_process_num_;

	nTuple<int, 3> m_topology_dims_;
	nTuple<int, 3> m_topology_strides_;
	nTuple<int, 3> m_topology_coord_;

	MPI_Comm m_comm_;

	void init(int argc, char** argv);

	int get_neighbour(int direction, int disp) const;
	nTuple<int, 3> get_coordinate(int rank) const;

	void decompose(int ndims, size_t *count, size_t * offset) const;
};

void MPIComm::pimpl_s::init(int argc, char** argv)
{

	m_comm_ = MPI_COMM_WORLD;

	m_num_process_ = 1;
	m_process_num_ = 0;
	m_topology_dims_ = 1;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(m_comm_, &m_num_process_);
	MPI_Comm_rank(m_comm_, &m_process_num_);

	m_topology_dims_[0] = m_num_process_;

	bool show_help = false;

	parse_cmd_line(argc, argv,

	[&](std::string const & opt,std::string const & value)->int
	{
		if( opt=="number_of_threads")
		{
//			string_to_value(value,&m_num_threads_);
		}
		else if( opt=="mpi_topology")
		{
			string_to_value(value,&m_topology_dims_);

		}
		else if( opt=="h" || opt=="help")
		{
			show_help=true;
		}

		return CONTINUE;

	}

	);

	if (show_help)
	{
		SHOW_OPTIONS("--number_of_threads <NUMBER>", "number of threads");
		SHOW_OPTIONS("--mpi_topology <NX,NY,NZ>",
				" set communicator's topology");
		return;
	}

	LOGGER.set_mpi_comm(m_process_num_, m_num_process_);

	if (NProduct(m_topology_dims_) != m_num_process_)
	{
		RUNTIME_ERROR("MPI topology is invalid!");
	}

	m_topology_strides_[0] = 1;
	m_topology_strides_[1] = m_topology_dims_[0];
	m_topology_strides_[2] = m_topology_dims_[1] * m_topology_strides_[1];

	m_topology_coord_ = get_coordinate(m_process_num_);

	VERBOSE << "MPI communicator is initialized!" << std::endl;
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
int MPIComm::pimpl_s::get_neighbour(int direction, int disp) const
{
	nTuple<int, 3> coord;
	coord = m_topology_coord_;
	coord[direction] = (coord[direction] + m_topology_dims_[direction] + disp)
			% m_topology_dims_[direction];
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

bool MPIComm::is_valid() const
{
	return pimpl_ != nullptr && pimpl_->m_comm_ != MPI_COMM_NULL;
}

int MPIComm::process_num() const
{
	return pimpl_->m_process_num_;
}

int MPIComm::num_of_process() const
{
	return pimpl_->m_num_process_;
}
void MPIComm::init(int argc, char** argv)
{
	if (pimpl_ == nullptr)
	{
		pimpl_ = std::unique_ptr<pimpl_s>(new pimpl_s);
	}
	pimpl_->init(argc, argv);
}

int MPIComm::get_neighbour(int direction, int disp)
{
	return pimpl_->get_neighbour(direction, disp);
}
nTuple<int, 3> const &MPIComm::get_topology() const
{
	return std::move(pimpl_->m_topology_dims_);
}
nTuple<int, 3> MPIComm::get_coordinate(int rank) const
{
	return std::move(pimpl_->get_coordinate(rank));
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
