/*
 * message_comm.h
 *
 *  Created on: 2014年5月12日
 *      Author: salmon
 */

#ifndef MESSAGE_COMM_H_
#define MESSAGE_COMM_H_

#include <mpi.h>
namespace simpla
{

class MessageComm
{
	int num_process_;
	int process_num_;
	MPI_Comm comm_;
public:
	MessageComm()
			: num_process_(1), process_num_(0), comm_(MPI_COMM_NULL)
	{
	}

	MessageComm(int argc, char** argv)
			: num_process_(1), process_num_(0), comm_(MPI_COMM_NULL)
	{
		Init(argc, argv);
	}
	~MessageComm()
	{
		Close();
	}

	void Init(int argc = 0, char** argv = nullptr)
	{
		MPI_Init(&argc, &argv);
		if (comm_ == MPI_COMM_NULL)
			comm_ = MPI_COMM_WORLD;

		MPI_Comm_size(comm_, &num_process_);
		MPI_Comm_rank(comm_, &process_num_);
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

}
;
#define GLOBAL_COMM   SingletonHolder<simpla::MessageComm>::instance()

}
// namespace simpla

#endif /* MESSAGE_COMM_H_ */
