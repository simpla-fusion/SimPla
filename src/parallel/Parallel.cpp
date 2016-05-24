/**
 * @file parallel.cpp
 *
 * @date    2014-7-28  AM11:11:49
 * @author salmon
 */

#include "../gtl/design_pattern/SingletonHolder.h"
#include "../gtl/parse_command_line.h"
#include "MPIComm.h"
#include "DistributedObject.h"

namespace simpla { namespace parallel
{

void init(int argc, char **argv)
{
    bool no_mpi = false;

    parse_cmd_line(argc, argv,

                   [&](std::string const &opt, std::string const &value) -> int
                   {
                       if (opt == "no-mpi") { no_mpi = true; }

                       return CONTINUE;
                   }

    );

    if (!no_mpi) { SingletonHolder<MPIComm>::instance().init(argc, argv); }
}

void close()
{
    SingletonHolder<MPIComm>::instance().close();
}

std::string help_message()
{
    return MPIComm::help_message();
};




///**
// * @param pos in {0,count} out {begin,shape}
// */
//template<typename Integral>
//std::tuple<Integral, Integral> sync_global_location(Integral count)
//{
//	Integral begin = 0;
//
//	if ( GLOBAL_COMM.is_valid() && GLOBAL_COMM.get_size() > 1)
//	{
//
//		auto communicator = GLOBAL_COMM.comm();
//
//		int num_of_process = GLOBAL_COMM.get_size();
//		int porcess_number = GLOBAL_COMM.get_rank();
//
//		MPIDataType m_type;
//
//		std::vector<Integral> buffer;
//
//		if (porcess_number == 0)
//		buffer.resize(num_of_process);
//
//		MPI_Gather(&count, 1, m_type.type(), &buffer[0], 1, m_type.type(), 0, communicator);
//
//		MPI_Barrier(communicator);
//
//		if (porcess_number == 0)
//		{
//			for (int i = 1; i < num_of_process; ++i)
//			{
//				buffer[i] += buffer[i - 1];
//			}
//			buffer[0] = count;
//			count = buffer[num_of_process - 1];
//
//			for (int i = num_of_process - 1; i > 0; --i)
//			{
//				buffer[i] = buffer[i - 1];
//			}
//			buffer[0] = 0;
//		}
//		MPI_Barrier(communicator);
//		MPI_Scatter(&buffer[0], 1, m_type.type(), &begin, 1, m_type.type(), 0, communicator);
//		MPI_Bcast(&count, 1, m_type.type(), 0, communicator);
//	}
//
//	return std::make_tuple(begin, count);
//
//}
//inline MPI_Op get_MPI_Op(std::string const & op_c)
//{
//	MPI_Op op = MPI_SUM;
//
//	if (op_c == "Max")
//	{
//		op = MPI_MAX;
//	}
//	else if (op_c == "Min")
//	{
//		op = MPI_MIN;
//	}
//	else if (op_c == "Sum")
//	{
//		op = MPI_SUM;
//	}
//	else if (op_c == "Prod")
//	{
//		op = MPI_PROD;
//	}
//	else if (op_c == "LAND")
//	{
//		op = MPI_LAND;
//	}
//	else if (op_c == "LOR")
//	{
//		op = MPI_LOR;
//	}
//	else if (op_c == "BAND")
//	{
//		op = MPI_BAND;
//	}
//	else if (op_c == "Sum")
//	{
//		op = MPI_BOR;
//	}
//	else if (op_c == "Sum")
//	{
//		op = MPI_MAXLOC;
//	}
//	else if (op_c == "Sum")
//	{
//		op = MPI_MINLOC;
//	}
//	return op;
//}
} //  namespace parallel
} // namespace simpla
