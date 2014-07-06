/*
 * message_comm.h
 *
 *  Created on: 2014-5-12
 *      Author: salmon
 */

#ifndef MESSAGE_COMM_H_
#define MESSAGE_COMM_H_

#include <mpi.h>
#include <algorithm>
#include <thread>
#include "../utilities/parse_command_line.h"
#include "../utilities/utilities.h"

namespace simpla
{

class MessageComm
{
	int num_threads_;
	int num_process_;
	int process_num_;
	MPI_Comm comm_;
public:
	MessageComm()
			: num_process_(1), process_num_(0), comm_(MPI_COMM_NULL), num_threads_(1)
	{
	}

	MessageComm(int argc, char** argv)
			: num_process_(1), process_num_(0), comm_(MPI_COMM_NULL), num_threads_(1)
	{
		Init(argc, argv);
	}
	~MessageComm()
	{
		Close();
	}

	void Init(int argc = 0, char** argv = nullptr)
	{
		if (comm_ == MPI_COMM_NULL)
		{
			MPI_Init(&argc, &argv);
			if (comm_ == MPI_COMM_NULL)
				comm_ = MPI_COMM_WORLD;

			MPI_Comm_size(comm_, &num_process_);
			MPI_Comm_rank(comm_, &process_num_);
		}

		ParseCmdLine(argc, argv,

		[&](std::string const & opt,std::string const & value)->int
		{
			if( opt=="number_of_thread")
			{
				num_threads_ =ToValue<size_t>(value);
			}

			return CONTINUE;

		}

		);

	}
	void Close()
	{
		if (comm_ != MPI_COMM_NULL)
			MPI_Finalize();

		comm_ = MPI_COMM_NULL;
	}
	MPI_Comm GetComm()
	{
		return comm_;
	}
	MPI_Info GetInfo()
	{
		return MPI_INFO_NULL;
	}

	bool IsInitilized() const
	{
		return comm_ != MPI_COMM_NULL;
	}
	int GetRank() const
	{
		return process_num_;
	}
	int ProcessNum() const
	{
		return process_num_;
	}

	int GetSize() const
	{
		return num_process_;
	}
	int NumProcess() const
	{
		return num_process_;
	}

	void Barrier()
	{
		if (comm_ != MPI_COMM_NULL)
			MPI_Barrier(comm_);
	}

	void SetNumThread(int num)
	{
		int local_num_cpu = std::thread::hardware_concurrency();
		num_threads_ = std::min(num, local_num_cpu);
	}
	unsigned int GetNumThreads() const
	{
		return num_threads_;
	}
}
;
#define GLOBAL_COMM   SingletonHolder<simpla::MessageComm>::instance()

}
// namespace simpla

#endif /* MESSAGE_COMM_H_ */
