/**
 * @file parallel.cpp
 *
 * @date    2014-7-28  AM11:11:49
 * @author salmon
 */
#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/parse_command_line.h>

#ifdef MPI_FOUND
#include <simpla/design_pattern/SingletonHolder.h>
#include "MPIComm.h"
#endif

namespace simpla {
namespace parallel {

void init(int argc, char **argv) {
    bool no_mpi = false;
    parse_cmd_line(argc, argv, [&](std::string const &opt, std::string const &value) -> int {
        if (opt == "no-mpi") { no_mpi = true; }
        return CONTINUE;
    });
#ifdef MPI_FOUND
    if (!no_mpi) { SingletonHolder<MPIComm>::instance().init(argc, argv); }
#endif
}

void close() {
#ifdef MPI_FOUND
    SingletonHolder<MPIComm>::instance().close();
#endif
}

std::string help_message() {
//#ifdef MPI_FOUND
//    return MPIComm::help_message();
//#endif
    return "";
};

///**
// * @_fdtd_param pos in {0,size} out {begin,m_global_dims_}
// */
// template<typename Integral>
// std::tuple<Integral, Integral> sync_global_location(Integral size)
//{
//	Integral begin = 0;
//
//	if ( GLOBAL_COMM.isValid() && GLOBAL_COMM.get_size() > 1)
//	{
//
//		auto communicator = GLOBAL_COMM.comm();
//
//		int num_of_process = GLOBAL_COMM.get_size();
//		int porcess_number = GLOBAL_COMM.rank();
//
//		MPIDataType m_type;
//
//		std::vector<Integral> m_buffer;
//
//		if (porcess_number == 0)
//		m_buffer.resize(num_of_process);
//
//		MPI_Gather(&size, 1, m_type.type(), &m_buffer[0], 1, m_type.value_type_info(), 0, communicator);
//
//		MPI_Barrier(communicator);
//
//		if (porcess_number == 0)
//		{
//			for (int i = 1; i < num_of_process; ++i)
//			{
//				m_buffer[i] += m_buffer[i - 1];
//			}
//			m_buffer[0] = size;
//			size = m_buffer[num_of_process - 1];
//
//			for (int i = num_of_process - 1; i > 0; --i)
//			{
//				m_buffer[i] = m_buffer[i - 1];
//			}
//			m_buffer[0] = 0;
//		}
//		MPI_Barrier(communicator);
//		MPI_Scatter(&m_buffer[0], 1, m_type.type(), &begin, 1, m_type.value_type_info(), 0, communicator);
//		MPI_Bcast(&size, 1, m_type.value_type_info(), 0, communicator);
//	}
//
//	return std::make_tuple(begin, size);
//
//}
// inline MPI_Op get_MPI_Op(std::string const & op_c)
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
}  //  namespace parallel
}  // namespace simpla
