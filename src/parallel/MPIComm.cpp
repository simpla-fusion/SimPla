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
#include "../gtl/Log.h"


namespace simpla { namespace parallel
{

struct MPIComm::pimpl_s
{
    static constexpr int MAX_NUM_OF_DIMS = 3;

    MPI_Comm m_comm_;

    size_type m_object_id_count_;

    int m_topology_ndims_ = 2;

    int m_topology_dims_[MAX_NUM_OF_DIMS];
};
MPIComm::MPIComm()
    : pimpl_(new pimpl_s)
{
    pimpl_->m_comm_ = MPI_COMM_NULL;
    pimpl_->m_object_id_count_ = 0;
    pimpl_->m_topology_ndims_ = 2;
    for (int i = 0; i < pimpl_->m_topology_ndims_; ++i) { pimpl_->m_topology_dims_[i] = 0; }
}
MPIComm::MPIComm(int argc, char **argv)
    : MPIComm()
{
    init(argc, argv);
}

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
    MPI_ERROR(MPI_Cart_rank(pimpl_->m_comm_, d, &res));
    return res;
}

void MPIComm::init(int argc, char **argv)
{

    pimpl_->m_object_id_count_ = 0;

    MPI_ERROR(MPI_Init(&argc, &argv));

    int m_num_process_;

    MPI_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &m_num_process_));


    if (m_num_process_ <= 1)
    {
        pimpl_->m_comm_ = MPI_COMM_NULL;
        MPI_Finalize();

    }
    else
    {

        int MAX_NUM_OF_NEIGHBOURS = 27;

        int m_process_num_;

        int m_topology_coord_[3];

        int m_topology_num_of_neighbour_;

        int m_topology_neighbours_[MAX_NUM_OF_NEIGHBOURS];

        CHECK(m_num_process_);

        CHECK(pimpl_->m_topology_ndims_);

        for (int i = 0; i < m_topology_ndims_; ++i) { m_topology_dims_[i] = 0; }

        MPI_ERROR(MPI_Dims_create(m_num_process_, 2, m_topology_dims_));

        CHECK(m_topology_dims_[0]);
        CHECK(m_topology_dims_[1]);

        int periods[m_topology_ndims_];
        for (int i = 0; i < m_topology_ndims_; ++i) { periods[i] = true; }
        MPI_ERROR(MPI_Cart_create(MPI_COMM_WORLD, m_topology_ndims_,
                                  m_topology_dims_,
                                  periods, 0,
                                  &pimpl_->m_comm_));


//        pimpl_->m_topology_num_of_neighbour_ = 2 * pimpl_->m_topology_ndims_;
//
//
//        MPI_ERROR(MPI_Comm_rank(pimpl_->m_comm_, &pimpl_->m_process_num_));

        logger::set_mpi_comm(rank(), size());

        MPI_ERROR(MPI_Cart_coords(comm(), size(), m_topology_ndims_, m_topology_coord_));


        INFORM << "MPI communicator is initialized! "
            "[("
               << m_topology_coord_[0] << ","
               << m_topology_coord_[1] << ","
               << m_topology_coord_[2]
               << ")/("
               << m_topology_dims_[0] << ","
               << m_topology_dims_[1] << ","
               << m_topology_dims_[2]
               << ")]" << std::endl;
    }
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

    if (comm() != MPI_COMM_NULL)
    {
        int tope_type = MPI_CART;
        MPI_ERROR(MPI_Topo_test(comm(), &tope_type));
        if (tope_type == MPI_CART);
        {
            MPI_ERROR(MPI_Cartdim_get(comm(), mpi_topo_ndims));

            MPI_ERROR(MPI_Cart_get(comm(), *mpi_topo_ndims, mpi_topo_dims, periods, mpi_topo_coord));
        }
    }
    return SP_SUCCESS;
};


void MPIComm::close()
{
    if (pimpl_ != nullptr && pimpl_->m_comm_ != MPI_COMM_NULL)
    {
        INFORM << "MPI Communicator is closed!" << std::endl;

        MPI_ERROR(MPI_Finalize());

        pimpl_->m_comm_ = MPI_COMM_NULL;

    }

}

}}//namespace simpla{namespace parallel{
