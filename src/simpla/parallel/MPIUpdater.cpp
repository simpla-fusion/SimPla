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
    MPI_Comm comm;
    int ndims = 3;
    int mpi_topology_ndims = 0;
    int mpi_dims[3], mpi_periods[3], mpi_coords[3];
    index_box_type local_box_;

    index_tuple m_gw_{2, 2, 2};

    MPI_Datatype ele_type;
    int tag;

    index_box_type send_box[2];
    index_box_type recv_box[2];
    void *send_buffer[2];
    void *recv_buffer[2];
    int send_size[2];
    int recv_size[2];
    int m_direction_ = 0;
};
MPIUpdater::MPIUpdater() : m_pimpl_(new pimpl_s) {
    m_pimpl_->comm = GLOBAL_COMM.comm();
    if (m_pimpl_->comm != MPI_COMM_NULL) {
        int topo_type = MPI_CART;
        MPI_CALL(MPI_Topo_test(m_pimpl_->comm, &topo_type));
        if (topo_type == MPI_CART) {
            MPI_CALL(MPI_Cartdim_get(m_pimpl_->comm, &m_pimpl_->mpi_topology_ndims));
            ASSERT(m_pimpl_->mpi_topology_ndims <= m_pimpl_->ndims);
            MPI_CALL(MPI_Cart_get(m_pimpl_->comm, 3, m_pimpl_->mpi_dims, m_pimpl_->mpi_periods, m_pimpl_->mpi_coords));
        }
    }
};
MPIUpdater::~MPIUpdater() { TearDown(); };

void MPIUpdater::SetDirection(int d) { m_pimpl_->m_direction_ = d; }
int MPIUpdater::GetDirection() const { return m_pimpl_->m_direction_; }
void MPIUpdater::SetGhostWidth(index_tuple const &gw) { m_pimpl_->m_gw_ = gw; }
index_tuple MPIUpdater::GetGhostWidth() const { return m_pimpl_->m_gw_; }
void MPIUpdater::SetIndexBox(index_box_type const &idx_box) { m_pimpl_->local_box_ = idx_box; }
index_box_type MPIUpdater::GetIndexBox() const { return m_pimpl_->local_box_; }

bool MPIUpdater::isSetUp() const { return m_pimpl_->m_is_setup_; }
bool MPIUpdater::isEnable() const {
    return m_pimpl_->comm != MPI_COMM_NULL && GLOBAL_COMM.size() > 1 && m_pimpl_->mpi_dims[m_pimpl_->m_direction_] > 1;
}

void MPIUpdater::SetUp() {
    if (m_pimpl_->m_is_setup_ || m_pimpl_->mpi_dims[m_pimpl_->m_direction_] <= 1) { return; }
    m_pimpl_->m_is_setup_ = true;

    for (int i = 0; i < 2; ++i) {
        m_pimpl_->send_box[i] = m_pimpl_->local_box_;
        m_pimpl_->recv_box[i] = m_pimpl_->local_box_;
    }

    auto d = m_pimpl_->m_direction_;
    //    if (m_pimpl_->mpi_dims[d] == 1) {
    //        std::get<0>(m_pimpl_->send_box[0]) = 0;
    //        std::get<0>(m_pimpl_->send_box[1]) = 0;
    //        std::get<0>(m_pimpl_->recv_box[0]) = 0;
    //        std::get<0>(m_pimpl_->recv_box[1]) = 0;
    //        std::get<1>(m_pimpl_->send_box[0]) = 0;
    //        std::get<1>(m_pimpl_->send_box[1]) = 0;
    //        std::get<1>(m_pimpl_->recv_box[0]) = 0;
    //        std::get<1>(m_pimpl_->recv_box[1]) = 0;
    //
    //    } else {
    std::get<0>(m_pimpl_->send_box[0])[d] = std::get<0>(m_pimpl_->local_box_)[d];
    std::get<1>(m_pimpl_->send_box[0])[d] = std::get<0>(m_pimpl_->local_box_)[d] + m_pimpl_->m_gw_[d];

    std::get<0>(m_pimpl_->recv_box[0])[d] = std::get<0>(m_pimpl_->local_box_)[d] - m_pimpl_->m_gw_[d];
    std::get<1>(m_pimpl_->recv_box[0])[d] = std::get<0>(m_pimpl_->local_box_)[d];

    std::get<0>(m_pimpl_->send_box[1])[d] = std::get<1>(m_pimpl_->local_box_)[d] - m_pimpl_->m_gw_[d];
    std::get<1>(m_pimpl_->send_box[1])[d] = std::get<1>(m_pimpl_->local_box_)[d];

    std::get<0>(m_pimpl_->recv_box[1])[d] = std::get<1>(m_pimpl_->local_box_)[d];
    std::get<1>(m_pimpl_->recv_box[1])[d] = std::get<1>(m_pimpl_->local_box_)[d] + m_pimpl_->m_gw_[d];
    //    }
    for (int i = 0; i < d; ++i) {
        std::get<0>(m_pimpl_->send_box[0])[i] = std::get<0>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
        std::get<1>(m_pimpl_->send_box[0])[i] = std::get<1>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];

        std::get<0>(m_pimpl_->recv_box[0])[i] = std::get<0>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
        std::get<1>(m_pimpl_->recv_box[0])[i] = std::get<1>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];

        std::get<0>(m_pimpl_->send_box[1])[i] = std::get<0>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
        std::get<1>(m_pimpl_->send_box[1])[i] = std::get<1>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];

        std::get<0>(m_pimpl_->recv_box[1])[i] = std::get<0>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
        std::get<1>(m_pimpl_->recv_box[1])[i] = std::get<1>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];
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
        m_pimpl_->send_size[i] =
            static_cast<int>(std::get<1>(m_pimpl_->send_box[i])[0] - std::get<0>(m_pimpl_->send_box[i])[0]) *
            static_cast<int>(std::get<1>(m_pimpl_->send_box[i])[1] - std::get<0>(m_pimpl_->send_box[i])[1]) *
            static_cast<int>(std::get<1>(m_pimpl_->send_box[i])[2] - std::get<0>(m_pimpl_->send_box[i])[2]);

        m_pimpl_->recv_size[i] =
            static_cast<int>(std::get<1>(m_pimpl_->recv_box[i])[0] - std::get<0>(m_pimpl_->recv_box[i])[0]) *
            static_cast<int>(std::get<1>(m_pimpl_->recv_box[i])[1] - std::get<0>(m_pimpl_->recv_box[i])[1]) *
            static_cast<int>(std::get<1>(m_pimpl_->recv_box[i])[2] - std::get<0>(m_pimpl_->recv_box[i])[2]);

        if (m_pimpl_->send_buffer[i] != nullptr) {
            delete m_pimpl_->send_buffer[i];
            m_pimpl_->send_buffer[i] = nullptr;
        }
        if (m_pimpl_->recv_buffer[i] != nullptr) {
            delete m_pimpl_->recv_buffer[i];
            m_pimpl_->recv_buffer[i] = nullptr;
        }

        ASSERT(m_pimpl_->send_size[i] > 0);
        ASSERT(m_pimpl_->recv_size[i] > 0);
        m_pimpl_->send_buffer[i] = operator new(m_pimpl_->send_size[i] * ele_size);
        m_pimpl_->recv_buffer[i] = operator new(m_pimpl_->recv_size[i] * ele_size);
    }

    for (int i = 0; i < 2; ++i) {
        GetSendBuffer(i).reset(m_pimpl_->send_buffer[i], &std::get<0>(m_pimpl_->send_box[i])[0],
                               &std::get<1>(m_pimpl_->send_box[i])[0]);
        GetRecvBuffer(i).reset(m_pimpl_->recv_buffer[i], &std::get<0>(m_pimpl_->recv_box[i])[0],
                               &std::get<1>(m_pimpl_->recv_box[i])[0]);
        GetSendBuffer(i).Clear();
        GetRecvBuffer(i).Clear();
    }
}

void MPIUpdater::TearDown() {
    for (int i = 0; i < 2; ++i) {
        delete m_pimpl_->send_buffer[i];
        m_pimpl_->send_buffer[i] = nullptr;
        delete m_pimpl_->recv_buffer[i];
        m_pimpl_->recv_buffer[i] = nullptr;
    }
    m_pimpl_->m_is_setup_ = false;
}

void MPIUpdater::SetTag(int tag) { m_pimpl_->tag = tag; }

void MPIUpdater::Push(ArrayBase const &a) {
    if (!isEnable()) { return; }
    GetSendBuffer(0).CopyIn(a);
    GetSendBuffer(1).CopyIn(a);
}
void MPIUpdater::Pop(ArrayBase &a) const {
    if (!isEnable()) { return; }
    a.CopyIn(GetRecvBuffer(0));
    a.CopyIn(GetRecvBuffer(1));
}

void MPIUpdater::SendRecv() {
    if (!isEnable()) { return; }
    int left, right;

    MPI_CALL(MPI_Cart_shift(m_pimpl_->comm, m_pimpl_->m_direction_, 1, &left, &right));

    GLOBAL_COMM.barrier();
    MPI_CALL(MPI_Sendrecv(m_pimpl_->send_buffer[0], m_pimpl_->send_size[0], m_pimpl_->ele_type, left, m_pimpl_->tag,
                          m_pimpl_->recv_buffer[1], m_pimpl_->recv_size[1], m_pimpl_->ele_type, right, m_pimpl_->tag,
                          m_pimpl_->comm, MPI_STATUS_IGNORE));
    GLOBAL_COMM.barrier();
    MPI_CALL(MPI_Sendrecv(m_pimpl_->send_buffer[1], m_pimpl_->send_size[1], m_pimpl_->ele_type, right, m_pimpl_->tag,
                          m_pimpl_->recv_buffer[0], m_pimpl_->recv_size[0], m_pimpl_->ele_type, left, m_pimpl_->tag,
                          m_pimpl_->comm, MPI_STATUS_IGNORE));
    GLOBAL_COMM.barrier();
}

}  // namespace parallel {
}  // namespace simpla {