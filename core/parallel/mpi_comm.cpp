/*
 * mpi_comm.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include "mpi_comm.h"

namespace simpla
{

MPIComm::MPIComm() :
		num_process_(1), process_num_(0), comm_(MPI_COMM_NULL), num_threads_(1)
{
}

MPIComm::MPIComm(int argc, char** argv) :
		num_process_(1), process_num_(0), comm_(MPI_COMM_NULL), num_threads_(1)
{
	init(argc, argv);
}
MPIComm::~MPIComm()
{
	close();
}

void MPIComm::init(int argc, char** argv)
{
	if (comm_ == MPI_COMM_NULL)
	{
		MPI_Init(&argc, &argv);
		if (comm_ == MPI_COMM_NULL)
			comm_ = MPI_COMM_WORLD;

		MPI_Comm_size(comm_, &num_process_);
		MPI_Comm_rank(comm_, &process_num_);

		parse_cmd_line(argc, argv,

				[&](std::string const & opt,std::string const & value)->int
				{
					if( opt=="number_of_threads")
					{
						num_threads_ =ToValue<size_t>(value);
					}
					else if( opt=="h")
					{
						SHOW_OPTIONS ("--mt <NUMBER>", "number of threads");

						return TERMINATE;
					}

					return CONTINUE;

				}

				);

		LOGGER.set_mpi_comm(process_num_, num_process_);

	}

}
void MPIComm::close()
{
	if (comm_ != MPI_COMM_NULL)
		MPI_Finalize();

	comm_ = MPI_COMM_NULL;
}

MPI_Comm MPIComm::comm()
{
	init();

	return comm_;
}
MPI_Info MPIComm::info()
{
	return MPI_INFO_NULL;
}

bool MPIComm::is_valid() const
{
	return comm_ != MPI_COMM_NULL;
}
int MPIComm::get_rank() const
{
	return process_num_;
}
int MPIComm::process_num() const
{
	return process_num_;
}

int MPIComm::get_size() const
{
	return num_process_;
}
int MPIComm::num_of_process() const
{
	return num_process_;
}

void MPIComm::barrier()
{
	if (comm_ != MPI_COMM_NULL)
		MPI_Barrier(comm_);
}

void MPIComm::set_num_of_threads(int num)
{
	int local_num_cpu = std::thread::hardware_concurrency();
	num_threads_ = std::min(num, local_num_cpu);
}
unsigned int MPIComm::get_num_of_threads() const
{
	return num_threads_;
}

}  // namespace simpla
