//
// Created by salmon on 17-9-18.
//
#include "MPIUpdater.h"
#include <mpi.h>
#include "MPIComm.h"
#include "simpla/SIMPLA_config.h"
#include "simpla/utilities/macro.h"

namespace simpla {
namespace parallel {

struct MPIUpdater::pimpl_s {
    bool m_is_setup_ = false;
    int ndims = 3;
    int mpi_topology_ndims = 0;
    int mpi_dims[3] = {1, 1, 1};
    int mpi_periods[3] = {1, 1, 1};
    int mpi_coords[3] = {0, 0, 0};
    index_box_type local_box_{{0, 0, 0}, {1, 1, 1}};

    index_tuple m_gw_{2, 2, 2};

    MPI_Datatype ele_type;
    int tag = 0;
    int m_direction_ = 0;
    int left = 0, right = 0;
    int m_rank_ = 0;
    int m_coord_[3] = {0, 0, 0};
};
MPIUpdater::MPIUpdater() : m_pimpl_(new pimpl_s) {
    if (GLOBAL_COMM.size() <= 1) { return; }

    SP_CALL(GLOBAL_COMM.topology(&m_pimpl_->mpi_topology_ndims, m_pimpl_->mpi_dims, m_pimpl_->mpi_periods,
                                 m_pimpl_->mpi_coords));
};
MPIUpdater::~MPIUpdater() { TearDown(); };

void MPIUpdater::SetDirection(int d) { m_pimpl_->m_direction_ = d; }
int MPIUpdater::GetDirection() const { return m_pimpl_->m_direction_; }
void MPIUpdater::SetGhostWidth(index_tuple const &gw) { m_pimpl_->m_gw_ = gw; }
index_tuple MPIUpdater::GetGhostWidth() const { return m_pimpl_->m_gw_; }
void MPIUpdater::SetIndexBox(index_box_type const &idx_box) { m_pimpl_->local_box_ = idx_box; }
index_box_type MPIUpdater::GetIndexBox() const { return m_pimpl_->local_box_; }

bool MPIUpdater::isSetUp() const { return m_pimpl_->m_is_setup_; }

void MPIUpdater::SetUp() {
    if (m_pimpl_->m_is_setup_) { return; }

    m_pimpl_->m_is_setup_ = true;
    index_box_type send_box[2];
    index_box_type recv_box[2];

    for (auto &b : send_box) { b = m_pimpl_->local_box_; }
    for (auto &b : recv_box) { b = m_pimpl_->local_box_; }

    auto d = m_pimpl_->m_direction_;

    std::get<0>(send_box[0])[d] = std::get<0>(m_pimpl_->local_box_)[d];
    std::get<1>(send_box[0])[d] = std::get<0>(m_pimpl_->local_box_)[d] + m_pimpl_->m_gw_[d];

    std::get<0>(recv_box[0])[d] = std::get<0>(m_pimpl_->local_box_)[d] - m_pimpl_->m_gw_[d];
    std::get<1>(recv_box[0])[d] = std::get<0>(m_pimpl_->local_box_)[d];

    std::get<0>(send_box[1])[d] = std::get<1>(m_pimpl_->local_box_)[d] - m_pimpl_->m_gw_[d];
    std::get<1>(send_box[1])[d] = std::get<1>(m_pimpl_->local_box_)[d];

    std::get<0>(recv_box[1])[d] = std::get<1>(m_pimpl_->local_box_)[d];
    std::get<1>(recv_box[1])[d] = std::get<1>(m_pimpl_->local_box_)[d] + m_pimpl_->m_gw_[d];

    for (int i = 0; i < d; ++i) {
        std::get<0>(send_box[0])[i] = std::get<0>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
        std::get<1>(send_box[0])[i] = std::get<1>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];

        std::get<0>(recv_box[0])[i] = std::get<0>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
        std::get<1>(recv_box[0])[i] = std::get<1>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];

        std::get<0>(send_box[1])[i] = std::get<0>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
        std::get<1>(send_box[1])[i] = std::get<1>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];

        std::get<0>(recv_box[1])[i] = std::get<0>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
        std::get<1>(recv_box[1])[i] = std::get<1>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];
    }

    size_type ele_size = 0;
    if (value_type_info() == typeid(int)) {
        ele_size = sizeof(int);
        m_pimpl_->ele_type = MPI_INT;
    } else if (value_type_info() == typeid(double)) {
        ele_size = sizeof(double);
        m_pimpl_->ele_type = MPI_DOUBLE;
    } else if (value_type_info() == typeid(float)) {
        ele_size = sizeof(float);
        m_pimpl_->ele_type = MPI_FLOAT;
    } else if (value_type_info() == typeid(long)) {
        ele_size = sizeof(long);
        m_pimpl_->ele_type = MPI_LONG;
    } else if (value_type_info() == typeid(unsigned long)) {
        ele_size = sizeof(unsigned long);
        m_pimpl_->ele_type = MPI_UNSIGNED_LONG;
    } else {
        UNIMPLEMENTED;
    }

    for (int i = 0; i < 2; ++i) {
        GetSendBuffer(i).reset(send_box[i]);
        GetRecvBuffer(i).reset(recv_box[i]);
        GetSendBuffer(i).Clear();
        GetRecvBuffer(i).Clear();
    }
#ifdef MPI_FOUND
    if (GLOBAL_COMM.is_valid()) {
        MPI_CALL(MPI_Comm_rank(GLOBAL_COMM.comm(), &m_pimpl_->m_rank_));
        MPI_CALL(MPI_Cart_shift(GLOBAL_COMM.comm(), m_pimpl_->m_direction_, 1, &m_pimpl_->left, &m_pimpl_->right));
    }
#endif
}

void MPIUpdater::TearDown() { m_pimpl_->m_is_setup_ = false; }

void MPIUpdater::SetTag(int tag) { m_pimpl_->tag = tag; }

void MPIUpdater::Push(ArrayBase const &a) {
    GetSendBuffer(0).CopyIn(a);
    GetSendBuffer(1).CopyIn(a);
}
void MPIUpdater::Pop(ArrayBase &a) const {
    a.CopyIn(GetRecvBuffer(0));
    a.CopyIn(GetRecvBuffer(1));
}

void MPIUpdater::SendRecv() {
    if (m_pimpl_->left != m_pimpl_->m_rank_ && m_pimpl_->right != m_pimpl_->m_rank_) {

        GLOBAL_COMM.barrier();
        MPI_CALL(MPI_Sendrecv(GetSendBuffer(0).pointer(), static_cast<int>(GetSendBuffer(0).size()), m_pimpl_->ele_type,
                              m_pimpl_->left, m_pimpl_->tag,  //
                              GetRecvBuffer(1).pointer(), static_cast<int>(GetRecvBuffer(1).size()), m_pimpl_->ele_type,
                              m_pimpl_->right, m_pimpl_->tag,  //
                              GLOBAL_COMM.comm(), MPI_STATUS_IGNORE));
        GLOBAL_COMM.barrier();
        MPI_CALL(MPI_Sendrecv(GetSendBuffer(1).pointer(), static_cast<int>(GetSendBuffer(1).size()), m_pimpl_->ele_type,
                              m_pimpl_->right, m_pimpl_->tag,  //
                              GetRecvBuffer(0).pointer(), static_cast<int>(GetRecvBuffer(0).size()), m_pimpl_->ele_type,
                              m_pimpl_->left, m_pimpl_->tag,  //
                              GLOBAL_COMM.comm(), MPI_STATUS_IGNORE));
        GLOBAL_COMM.barrier();
    } else  // if (m_pimpl_->mpi_periods[m_pimpl_->m_direction_] > 0)
    {
        index_tuple shift = {0, 0, 0};
        shift[m_pimpl_->m_direction_] = std::get<1>(m_pimpl_->local_box_)[m_pimpl_->m_direction_] -
                                        std::get<0>(m_pimpl_->local_box_)[m_pimpl_->m_direction_];
        auto s0 = GetSendBuffer(0).DuplicateArray();
        s0->Shift(&shift[0]);
        GetRecvBuffer(1).CopyIn(*s0);
        shift[m_pimpl_->m_direction_] *= -1;
        auto s1 = GetSendBuffer(1).DuplicateArray();
        s1->Shift(&shift[0]);
        GetRecvBuffer(0).CopyIn(*s1);
    }
}

}  // namespace parallel {
}  // namespace simpla {