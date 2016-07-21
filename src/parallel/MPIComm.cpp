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

    static constexpr int MAX_NUM_OF_NEIGHBOURS = 27;

    MPI_Comm m_comm_;

    int m_num_process_;

    int m_process_num_;

    int m_topology_ndims_ = 2;

    int m_topology_dims_[MAX_NUM_OF_DIMS];

    int m_topology_coord_[MAX_NUM_OF_DIMS];

    int m_topology_num_of_neighbour_;

    int m_topology_neighbours_[MAX_NUM_OF_NEIGHBOURS];

    size_type m_object_id_count_;

};

MPIComm::MPIComm()
    : pimpl_(nullptr) { }

MPIComm::MPIComm(int argc, char **argv)
    : pimpl_(nullptr) { init(argc, argv); }

MPIComm::~MPIComm() { close(); }

int MPIComm::process_num() const { return (!pimpl_) ? 0 : pimpl_->m_process_num_; }

int MPIComm::num_of_process() const { return (!pimpl_) ? 1 : pimpl_->m_num_process_; }

int MPIComm::get_rank() const { return (!pimpl_) ? 0 : pimpl_->m_process_num_; }

int MPIComm::get_rank(int const *d) const
{
    int res = 0;
    MPI_ERROR(MPI_Cart_rank(pimpl_->m_comm_, d, &res));
    return res;
}

void MPIComm::init(int argc, char **argv)
{
    if (!pimpl_) { pimpl_ = std::unique_ptr<pimpl_s>(new pimpl_s); }

    pimpl_->m_object_id_count_ = 0;

    MPI_ERROR(MPI_Init(&argc, &argv));

    MPI_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &pimpl_->m_num_process_));

    MPI_ERROR(MPI_Dims_create(pimpl_->m_num_process_, pimpl_->m_topology_ndims_, &pimpl_->m_topology_dims_[0]));

    {
        int periods[pimpl_->m_topology_ndims_];
        for (int i = 0; i < pimpl_->m_topology_ndims_; ++i) { periods[i] = true; }
        MPI_ERROR(MPI_Cart_create(MPI_COMM_WORLD, pimpl_->m_topology_ndims_,
                                  pimpl_->m_topology_dims_,
                                  periods, 0,
                                  &pimpl_->m_comm_));


        pimpl_->m_topology_num_of_neighbour_ = 2 * pimpl_->m_topology_ndims_;
    }

    MPI_ERROR(MPI_Comm_rank(pimpl_->m_comm_, &pimpl_->m_process_num_));

    logger::set_mpi_comm(pimpl_->m_process_num_, pimpl_->m_num_process_);

    MPI_ERROR(MPI_Cart_coords(pimpl_->m_comm_,
                              pimpl_->m_process_num_,
                              pimpl_->m_topology_ndims_,
                              pimpl_->m_topology_coord_));


    INFORM << "MPI communicator is initialized! "
        "[("
        << pimpl_->m_topology_coord_[0] << ","
        << pimpl_->m_topology_coord_[1] << ","
        << pimpl_->m_topology_coord_[2]
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

MPI_Comm MPIComm::comm() { return (!pimpl_) ? MPI_COMM_NULL : pimpl_->m_comm_; }

MPI_Info MPIComm::info()
{
    assert(pimpl_ != nullptr);
    return MPI_INFO_NULL;
}

void MPIComm::barrier() { if (!!pimpl_) MPI_Barrier(comm()); }

bool MPIComm::is_valid() const { return ((!!pimpl_) && pimpl_->m_comm_ != MPI_COMM_NULL) && num_of_process() > 1; }

int MPIComm::topology_num_of_dims() const { return (!pimpl_) ? 0 : pimpl_->m_topology_ndims_; };

void MPIComm::topology_num_of_dims(int n) { if (!pimpl_) { pimpl_->m_topology_ndims_ = n; }}

int const *MPIComm::topology_dims() const { return (!pimpl_) ? nullptr : (&pimpl_->m_topology_dims_[0]); }

void MPIComm::topology_coordinate(int rank, int *coord) const
{
    if (pimpl_ != nullptr && rank > 0 && coord == nullptr)
    {
        MPI_ERROR(MPI_Cart_coords(pimpl_->m_comm_, rank, pimpl_->m_topology_ndims_, coord));
    }
}
int MPIComm::topology_num_of_neighbours() const { return (pimpl_ == nullptr) ? 0 : pimpl_->m_topology_num_of_neighbour_; };

int const *MPIComm::topology_neighbours() const { return (pimpl_ == nullptr) ? nullptr : pimpl_->m_topology_neighbours_; };

int MPIComm::topology_neighbour(const int *d) const
{
    assert(pimpl_ != nullptr);
    int src = get_rank(), dest = src;

    for (int i = 0; i < pimpl_->m_topology_ndims_; ++i)
    {
        MPI_ERROR(MPI_Cart_shift(pimpl_->m_comm_, i, d[i], &src, &dest));
    }

    return dest;

}

void MPIComm::close()
{
    if (!!pimpl_ && pimpl_->m_comm_ != MPI_COMM_NULL)
    {
        INFORM << "MPI Communicator is closed!" << std::endl;

        MPI_ERROR(MPI_Finalize());

        pimpl_->m_comm_ = MPI_COMM_NULL;

     }

}

}}//namespace simpla{namespace parallel{
