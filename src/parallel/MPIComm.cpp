/**
 * @file mpi_comm.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#include "MPIComm.h"
#include <iostream>
#include "../sp_def.h"
#include "../gtl/Log.h"


namespace simpla { namespace parallel
{
struct MPIComm::pimpl_s
{
    MPI_Comm m_comm_;

    int m_num_process_;

    int m_process_num_;

    int m_topology_dims_[NDIMS];

    int m_topology_coord_[NDIMS];

    size_type m_object_id_count_;

};

constexpr int MPIComm::NDIMS;

MPIComm::MPIComm()
    : pimpl_(nullptr) { }

MPIComm::MPIComm(int argc, char **argv)
    : pimpl_(nullptr) { init(argc, argv); }

MPIComm::~MPIComm() { close(); }

int MPIComm::process_num() const { return (!pimpl_) ? 0 : pimpl_->m_process_num_; }

int MPIComm::num_of_process() const { return (!pimpl_) ? 1 : pimpl_->m_num_process_; }

void MPIComm::init(int argc, char **argv)
{
    if (!pimpl_) { pimpl_ = std::unique_ptr<pimpl_s>(new pimpl_s); }

    pimpl_->m_object_id_count_ = 0;

    MPI_ERROR(MPI_Init(&argc, &argv));

    MPI_ERROR(MPI_Comm_size(pimpl_->m_comm_, &pimpl_->m_num_process_));

    MPI_ERROR(MPI_Comm_rank(pimpl_->m_comm_, &pimpl_->m_process_num_));

    logger::set_mpi_comm(pimpl_->m_process_num_, pimpl_->m_num_process_);


    MPI_ERROR(MPI_Dims_create(pimpl_->m_num_process_, NDIMS, &pimpl_->m_topology_dims_[0]));

    {
        int periods[NDIMS];
        for (int i = 0; i < NDIMS; ++i) { periods[i] = true; }
        MPI_ERROR(MPI_Cart_create(MPI_COMM_WORLD, NDIMS,
                                  pimpl_->m_topology_dims_,
                                  periods, 0,
                                  &pimpl_->m_comm_));
    }

    MPI_ERROR(MPI_Cart_coords(pimpl_->m_comm_, pimpl_->m_process_num_, NDIMS, pimpl_->m_topology_coord_));


    VERBOSE << "MPI communicator is initialized!" << std::endl;
}

size_type MPIComm::generate_object_id()
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

MPI_Comm MPIComm::comm() { return (!pimpl_) ? MPI_COMM_NULL : pimpl_->m_comm_; }

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

void MPIComm::barrier() { if (!!pimpl_) MPI_Barrier(comm()); }

bool MPIComm::is_valid() const { return ((!!pimpl_) && pimpl_->m_comm_ != MPI_COMM_NULL) && num_of_process() > 1; }

void MPIComm::coordinate(int rank, int *coord) const
{
    if (pimpl_ != nullptr && rank > 0 && coord == nullptr)
    {
        MPI_ERROR(MPI_Cart_coords(pimpl_->m_comm_, rank, NDIMS, coord));
    }
}
int MPIComm::get_num_neighbours() const
{
    int num = 1;
    for (int i = 0; i < NDIMS; ++i) { num *= pimpl_->m_topology_dims_[i] > 0 ? 3 : 1; }
    return num - 1;
};
int MPIComm::get_neighbour(const int *d) const
{
    if (pimpl_ != nullptr)
    {
        int coord[NDIMS];
        for (int i = 0; i < NDIMS; ++i)
        {
            coord[i] = pimpl_->m_topology_coord_[i] + d[i];
        }
        return get_rank(coord);
    }
    else
    {
        return -1;
    }

}

void MPIComm::close()
{
    if (!!pimpl_ && pimpl_->m_comm_ != MPI_COMM_NULL)
    {
        MPI_ERROR(MPI_Finalize());

        pimpl_->m_comm_ = MPI_COMM_NULL;

        VERBOSE << "MPI Communicator is closed!" << std::endl;
    }

}

int const *MPIComm::dims() const { return (!pimpl_) ? nullptr : (&pimpl_->m_topology_dims_[0]); }

int MPIComm::get_rank() const { return (!pimpl_) ? 0 : pimpl_->m_process_num_; }

int MPIComm::get_rank(int const *d) const
{
    int res = 0;
    MPI_ERROR(MPI_Cart_rank(pimpl_->m_comm_, d, &res));
    return res;
}

}}//namespace simpla{namespace parallel{
