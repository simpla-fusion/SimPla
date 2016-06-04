/**
 * @file mpi_comm.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#include "MPIComm.h"

#include <stddef.h>
#include <cstdbool>
#include <iostream>
#include <memory>
#include <string>

#include "../gtl/Utilities.h"

namespace simpla { namespace parallel
{
struct MPIComm::pimpl_s
{
    MPI_Comm m_comm_;

    int m_num_process_;

    int m_process_num_;

    int m_object_id_count_;

    nTuple<int, NDIMS> m_topology_dims_;

    nTuple<int, NDIMS> m_topology_strides_;

    nTuple<int, NDIMS> m_topology_coord_;

};

constexpr int MPIComm::NDIMS;

MPIComm::MPIComm()
        : pimpl_(nullptr)
{
}

MPIComm::MPIComm(int argc, char **argv)
        : pimpl_(nullptr)
{
    init(argc, argv);
}

MPIComm::~MPIComm()
{
    close();
}

int MPIComm::process_num() const
{
    return (!pimpl_) ? 0 : pimpl_->m_process_num_;

}

int MPIComm::num_of_process() const
{
    return (!pimpl_) ? 1 : pimpl_->m_num_process_;
}

void MPIComm::init(int argc, char **argv)
{
    if (!pimpl_) { pimpl_ = std::unique_ptr<pimpl_s>(new pimpl_s); }

    pimpl_->m_comm_ = MPI_COMM_WORLD;

    pimpl_->m_num_process_ = 1;
    pimpl_->m_process_num_ = 0;
    pimpl_->m_topology_dims_ = 1;
    pimpl_->m_object_id_count_ = 0;

    pimpl_->m_topology_dims_ = 1;
    pimpl_->m_topology_coord_ = 0;
    pimpl_->m_topology_strides_ = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(pimpl_->m_comm_, &pimpl_->m_num_process_);
    MPI_Comm_rank(pimpl_->m_comm_, &pimpl_->m_process_num_);

    logger::set_mpi_comm(pimpl_->m_process_num_, pimpl_->m_num_process_);

    topology(nTuple<int, 3>({pimpl_->m_num_process_, 1, 1}));

    parse_cmd_line(argc, argv,

                   [&](std::string const &opt, std::string const &value) -> int
                   {

//		if( opt=="number_of_threads")
//		{
////			string_to_value(value,&m_num_threads_);
//		}
//		else

                       if (opt == "mpi_topology")
                       {
                           topology(type_cast<nTuple<int, 3 >>(value));
                       }

                       return CONTINUE;

                   }

    );

    VERBOSE << "MPI communicator is initialized!" << std::endl;
}

std::string MPIComm::help_message()
{
    return
        //"\t--number_of_threads <NUMBER>  \t, Number of threads \n"
            "\t--mpi_topology <NX NY NZ>    \t, Set Topology of mpi communicator. \n";

}

int MPIComm::generate_object_id()
{
    if (!pimpl_)
    {
        return 0;
    }
    else
    {
        ++pimpl_->m_object_id_count_;
        return pimpl_->m_object_id_count_;
    }
}

MPI_Comm MPIComm::comm()
{
    return (!pimpl_) ? MPI_COMM_NULL : pimpl_->m_comm_;
}

MPI_Info MPIComm::info()
{
    if (!pimpl_)
    {
        return MPI_INFO_NULL;
    }
    else
    {
        return MPI_INFO_NULL;
    }

}

void MPIComm::barrier()
{
    if (!!pimpl_)
        MPI_Barrier(comm());
}

bool MPIComm::is_valid() const
{
    return ((!!pimpl_) && pimpl_->m_comm_ != MPI_COMM_NULL)
           && num_of_process() > 1;
}

std::tuple<int, int, int> MPIComm::make_send_recv_tag(size_t prefix,
                                                      const nTuple<ptrdiff_t, 3> &offset)
{

    int dest_id = get_neighbour(offset);

    int send_tag = static_cast<int>(prefix << (NDIMS * 2));

    int recv_tag = static_cast<int>(prefix << (NDIMS * 2));

    for (int i = 0; i < NDIMS; ++i)
    {
        send_tag |= ((offset[i] + 1) & 3UL) << (2UL * i);

        recv_tag |= ((-offset[i] + 1) & 3UL) << (2UL * i);
    }
    return std::make_tuple(dest_id, send_tag, recv_tag);
}


nTuple<int, 3> MPIComm::coordinate(int rank) const
{
    if (!pimpl_)
    {
        return nTuple<int, 3>({0, 0, 0});
    }

    if (rank < 0)
    {
        rank = pimpl_->m_process_num_;
    }
    nTuple<int, 3> coord;
    coord[2] = rank / pimpl_->m_topology_strides_[2];
    coord[1] = (rank - (coord[2] * pimpl_->m_topology_strides_[2]))
               / pimpl_->m_topology_strides_[1];
    coord[0] = rank % pimpl_->m_topology_dims_[0];
    return std::move(coord);
}

int MPIComm::get_neighbour(nTuple<ptrdiff_t, 3> const &d) const
{

    return (!pimpl_) ? 0 : get_rank(pimpl_->m_topology_coord_ + d);

//			(((m_pimpl_->m_topology_coord_[0] + m_pimpl_->m_topology_dims_[0] + d[0])
//					% m_pimpl_->m_topology_dims_[0])
//					* m_pimpl_->m_topology_strides_[0]
//					+ ((m_pimpl_->m_topology_coord_[1]
//
//					+ m_pimpl_->m_topology_dims_[1] + d[1])
//							% m_pimpl_->m_topology_dims_[1])
//							* m_pimpl_->m_topology_strides_[1]
//
//					+ ((m_pimpl_->m_topology_coord_[2]
//							+ m_pimpl_->m_topology_dims_[2] + d[2])
//							% m_pimpl_->m_topology_dims_[2])
//							* m_pimpl_->m_topology_strides_[2]
//
//			);
    ;
}

void MPIComm::close()
{
    if (!!pimpl_ && pimpl_->m_comm_ != MPI_COMM_NULL)
    {
        MPI_Finalize();

        pimpl_->m_comm_ = MPI_COMM_NULL;

        VERBOSE << "MPI Communicator is closed!" << std::endl;
    }

}

void MPIComm::topology(nTuple<int, 3> const &d)
{
    if (!pimpl_)
        return;

    if (d[0] * d[1] * d[2] != pimpl_->m_num_process_)
    {
        VERBOSE << " processor number=" << pimpl_->m_num_process_ <<
        " topology=" << d << std::endl;
        THROW_EXCEPTION_RUNTIME_ERROR("MPI Topology is invalid!");
    }

    pimpl_->m_topology_dims_ = d;

    pimpl_->m_topology_strides_[0] = 1;
    pimpl_->m_topology_strides_[1] = pimpl_->m_topology_dims_[0];
    pimpl_->m_topology_strides_[2] = pimpl_->m_topology_dims_[1]
                                     * pimpl_->m_topology_strides_[1];

    pimpl_->m_topology_coord_ = coordinate(pimpl_->m_process_num_);
}

nTuple<int, 3> MPIComm::topology() const
{

    return (!pimpl_) ? nTuple<int, 3>({1, 1, 1}) : (pimpl_->m_topology_dims_);
}

int MPIComm::get_rank() const
{

    return (!pimpl_) ? 0 : pimpl_->m_process_num_;

}

int MPIComm::get_rank(nTuple<int, 3> const &d) const
{

    return (!pimpl_) ?
           0 :
           (

                   ((d[0] + pimpl_->m_topology_dims_[0]) % pimpl_->m_topology_dims_[0])
                   * pimpl_->m_topology_strides_[0]

                   + ((d[1] + pimpl_->m_topology_dims_[1])
                      % pimpl_->m_topology_dims_[1])
                     * pimpl_->m_topology_strides_[1]

                   + ((d[2] + pimpl_->m_topology_dims_[2])
                      % pimpl_->m_topology_dims_[2])
                     * pimpl_->m_topology_strides_[2]

           );
}

}}//namespace simpla{namespace parallel{
