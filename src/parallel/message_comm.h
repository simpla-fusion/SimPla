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
public:
	MessageComm()
			: num_process_(1), process_num_(0), isInitilized_(false)
	{
	}

	MessageComm(int argc, char** argv)
			: isInitilized_(false)
	{
		Init(argc, argv);
	}
	~MessageComm()
	{
		if (isInitilized_)
			MPI_Finalize();
	}

	void Init(int argc, char** argv)
	{
		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &num_process_);
		MPI_Comm_rank(MPI_COMM_WORLD, &process_num_);
		isInitilized_ = true;
	}
	MPI_Comm GetComm()
	{
		return MPI_COMM_WORLD;
	}
	MPI_Info GetInfo()
	{
		return MPI_INFO_NULL;
	}

	bool IsInitilized() const
	{
		return isInitilized_;
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
private:
	bool isInitilized_;
};
#define GLOBAL_COMM   SingletonHolder<simpla::MessageComm>::instance()

}
// namespace simpla

#endif /* MESSAGE_COMM_H_ */
