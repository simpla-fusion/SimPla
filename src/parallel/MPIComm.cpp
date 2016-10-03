/**
 * @file mpi_comm.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#include "MPIComm.h"
#include <iostream>
#include <cassert>
#include "../sp_def.h"
#include "../toolbox/Log.h"


namespace simpla { namespace parallel
{

struct MPIComm::pimpl_s
{
    static constexpr int MAX_NUM_OF_DIMS = 3;

    MPI_Comm m_comm_ = MPI_COMM_NULL;

    size_type m_object_id_count_ = 0;

    int m_topology_ndims_ = 3;

    int m_topology_dims_[3] = {0, 0, 0};
};

MPIComm::MPIComm()
        : pimpl_(new pimpl_s) {}

MPIComm::MPIComm(int argc, char **argv)
        : MPIComm() { init(argc, argv); }

MPIComm::~MPIComm() { close(); }

int MPIComm::process_num() const { return rank(); }

int MPIComm::num_of_process() const { return size(); }

int MPIComm::rank() const
{
    int res = 0;
    if (comm() != MPI_COMM_NULL) { MPI_Comm_rank(comm(), &res); }
    return res;
}

int MPIComm::size() const
{
    int res = 1;
    if (comm() != MPI_COMM_NULL)
    {
        MPI_Comm_size(comm(), &res);
    }
    return res;
}

int MPIComm::get_rank(int const *d) const
{
    int res = 0;
    MPI_CALL(MPI_Cart_rank(pimpl_->m_comm_, (int *) d, &res));
    return res;
}

void MPIComm::init(int argc, char **argv)
{

    pimpl_->m_object_id_count_ = 0;

    MPI_CALL(MPI_Init(&argc, &argv));

    int m_num_process_;

    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &m_num_process_));

    int m_topology_coord_[3] = {0, 0, 0};

    MPI_CALL(MPI_Dims_create(m_num_process_, pimpl_->m_topology_ndims_, pimpl_->m_topology_dims_));

    int periods[pimpl_->m_topology_ndims_];

    for (int i = 0; i < pimpl_->m_topology_ndims_; ++i) { periods[i] = true; }

    MPI_CALL(MPI_Cart_create(MPI_COMM_WORLD, pimpl_->m_topology_ndims_,
                             pimpl_->m_topology_dims_, periods, MPI_ORDER_C, &pimpl_->m_comm_));


    logger::set_mpi_comm(rank(), size());

    MPI_CALL(MPI_Cart_coords(comm(), rank(), pimpl_->m_topology_ndims_, m_topology_coord_));


    INFORM << "MPI communicator is initialized! "
            "[("
           << m_topology_coord_[0] << ","
           << m_topology_coord_[1] << ","
           << m_topology_coord_[2]
           << ")/("
           << pimpl_->m_topology_dims_[0] << ","
           << pimpl_->m_topology_dims_[1] << ","
           << pimpl_->m_topology_dims_[2]
           << ")]" << std::endl;

}

size_type MPIComm::generate_object_id()
{
    assert(pimpl_ != nullptr);

    ++(pimpl_->m_object_id_count_);

    return pimpl_->m_object_id_count_;

}

MPI_Comm MPIComm::comm() const { return (!pimpl_) ? MPI_COMM_NULL : pimpl_->m_comm_; }

MPI_Info MPIComm::info()
{
    assert(pimpl_ != nullptr);
    return MPI_INFO_NULL;
}

void MPIComm::barrier() { if (comm() != MPI_COMM_NULL) { MPI_Barrier(comm()); }}

bool MPIComm::is_valid() const { return ((!!pimpl_) && pimpl_->m_comm_ != MPI_COMM_NULL) && num_of_process() > 1; }

int MPIComm::topology(int *mpi_topo_ndims, int *mpi_topo_dims, int *periods, int *mpi_topo_coord) const
{
    *mpi_topo_ndims = 0;

    if (comm() == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    int tope_type = MPI_CART;

    MPI_CALL(MPI_Topo_test(comm(), &tope_type));

    if (tope_type == MPI_CART);
    {
        MPI_CALL(MPI_Cartdim_get(comm(), mpi_topo_ndims));

        MPI_CALL(MPI_Cart_get(comm(), *mpi_topo_ndims, mpi_topo_dims, periods, mpi_topo_coord));
    }

    return SP_SUCCESS;
};


void MPIComm::close()
{
    if (pimpl_ != nullptr && pimpl_->m_comm_ != MPI_COMM_NULL)
    {
        VERBOSE << "MPI Communicator is closed!" << std::endl;

        MPI_CALL(MPI_Finalize());

        pimpl_->m_comm_ = MPI_COMM_NULL;

    }

}

}}//namespace simpla{namespace parallel{
