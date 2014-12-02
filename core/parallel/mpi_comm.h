/*
 * mpi_comm.h
 *
 *  created on: 2014-5-12
 *      Author: salmon
 */

#ifndef MPI_COMM_H_
#define MPI_COMM_H_

extern "C"
{
#include <mpi.h>
}
#include <algorithm>
#include <thread>
#include "../utilities/parse_command_line.h"
#include "../utilities/misc_utilities.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/log.h"

namespace simpla
{

class MPIComm
{
	int num_threads_;
	int num_process_;
	int process_num_;
	MPI_Comm comm_;

	bool no_mpi_ = false;
public:
	MPIComm();

	MPIComm(int argc, char** argv);

	~MPIComm();

	void init(int argc = 0, char** argv = nullptr);

	void close();

	MPI_Comm comm();
	MPI_Info info();

	bool is_valid() const;
	int get_rank() const;
	int process_num() const;
	int get_size() const;
	int num_of_process() const;
	void barrier();
	void set_num_of_threads(int num);
	unsigned int get_num_of_threads() const;
}
;
#define GLOBAL_COMM   SingletonHolder<simpla::MPIComm>::instance()

}
// namespace simpla

#endif /* MPI_COMM_H_ */
