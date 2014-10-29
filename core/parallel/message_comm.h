/*
 * message_comm.h
 *
 *  created on: 2014-5-12
 *      Author: salmon
 */

#ifndef MESSAGE_COMM_H_
#define MESSAGE_COMM_H_
extern "C"
{
#include <mpi.h>
}
#include <algorithm>
#include <thread>
#include "../utilities/parse_command_line.h"
#include "../utilities/utilities.h"
#include "../utilities/singleton_holder.h"

namespace simpla
{

class MessageComm
{
	int num_threads_;
	int num_process_;
	int process_num_;
	MPI_Comm comm_;
public:
	MessageComm() :
			num_process_(1), process_num_(0), comm_(MPI_COMM_NULL), num_threads_(
					1)
	{
	}

	MessageComm(int argc, char** argv) :
			num_process_(1), process_num_(0), comm_(MPI_COMM_NULL), num_threads_(
					1)
	{
		init(argc, argv);
	}
	~MessageComm()
	{
		close();
	}

	void init(int argc = 0, char** argv = nullptr)
	{
		if (comm_ == MPI_COMM_NULL)
		{
			MPI_Init(&argc, &argv);
			if (comm_ == MPI_COMM_NULL)
				comm_ = MPI_COMM_WORLD;

			MPI_Comm_size(comm_, &num_process_);
			MPI_Comm_rank(comm_, &process_num_);

			ParseCmdLine(argc, argv,

			[&](std::string const & opt,std::string const & value)->int
			{
				if( opt=="number_of_threads")
				{
					num_threads_ =ToValue<size_t>(value);
				}

				return CONTINUE;

			}

			);

		}

	}
	void close()
	{
		if (comm_ != MPI_COMM_NULL)
			MPI_Finalize();

		comm_ = MPI_COMM_NULL;
	}
	MPI_Comm comm()
	{
		init();

		return comm_;
	}
	MPI_Info info()
	{
		return MPI_INFO_NULL;
	}

	bool is_ready() const
	{
		return comm_ != MPI_COMM_NULL;
	}
	int get_rank() const
	{
		return process_num_;
	}
	int process_num() const
	{
		return process_num_;
	}

	int get_size() const
	{
		return num_process_;
	}
	int num_of_process() const
	{
		return num_process_;
	}

	void barrier()
	{
		if (comm_ != MPI_COMM_NULL)
			MPI_Barrier(comm_);
	}

	void set_num_of_threads(int num)
	{
		int local_num_cpu = std::thread::hardware_concurrency();
		num_threads_ = std::min(num, local_num_cpu);
	}
	unsigned int get_num_of_threads() const
	{
		return num_threads_;
	}
}
;
#define GLOBAL_COMM   SingletonHolder<simpla::MessageComm>::instance()

}
// namespace simpla

#endif /* MESSAGE_COMM_H_ */
