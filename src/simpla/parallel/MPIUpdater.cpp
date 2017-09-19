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
    index_box_type send_box[6];
    index_box_type recv_box[6];
    index_tuple m_gw_{2, 2, 2};

    MPI_Datatype ele_type;
    int tag;
    void *send_buffer[6];
    void *recv_buffer[6];
    int send_size[6];
    int recv_size[6];
};
MPIUpdater::MPIUpdater() : m_pimpl_(new pimpl_s) {
    if (GLOBAL_COMM.comm() == MPI_COMM_NULL) { return; }

    m_pimpl_->comm = GLOBAL_COMM.comm();
    if (GLOBAL_COMM.comm() == MPI_COMM_NULL) { return; }
    int tope_type = MPI_CART;
    MPI_CALL(MPI_Topo_test(GLOBAL_COMM.comm(), &tope_type));
    if (tope_type != MPI_CART) { return; }
    int tag = 0;
    MPI_CALL(MPI_Cartdim_get(GLOBAL_COMM.comm(), &m_pimpl_->mpi_topology_ndims));
};
MPIUpdater::~MPIUpdater() { TearDown(); };
void MPIUpdater::SetGhostWidth(index_tuple const &gw) { m_pimpl_->m_gw_ = gw; }
index_tuple MPIUpdater::GetGhostWidth() const { return m_pimpl_->m_gw_; }
void MPIUpdater::SetIndexBox(index_box_type const &idx_box) { m_pimpl_->local_box_ = idx_box; }
index_box_type MPIUpdater::GetIndexBox() const { return m_pimpl_->local_box_; }

bool MPIUpdater::isSetUp() const { return m_pimpl_->m_is_setup_; }
bool MPIUpdater::isEnable() const { return m_pimpl_->comm != MPI_COMM_NULL && GLOBAL_COMM.size() > 1; }

void MPIUpdater::SetUp() {
    m_pimpl_->m_is_setup_ = true;
    int topo_type = MPI_CART;

    MPI_CALL(MPI_Topo_test(m_pimpl_->comm, &topo_type));

    ASSERT(topo_type == MPI_CART);

    MPI_CALL(MPI_Cartdim_get(m_pimpl_->comm, &m_pimpl_->mpi_topology_ndims));

    ASSERT(m_pimpl_->mpi_topology_ndims <= m_pimpl_->ndims);

    MPI_CALL(MPI_Cart_get(m_pimpl_->comm, 3, m_pimpl_->mpi_dims, m_pimpl_->mpi_periods, m_pimpl_->mpi_coords));
    for (int i = 0; i < 6; ++i) {
        m_pimpl_->send_box[i] = m_pimpl_->local_box_;
        m_pimpl_->recv_box[i] = m_pimpl_->local_box_;
    }

    for (int d = 0; d < m_pimpl_->mpi_topology_ndims; ++d) {
        if (m_pimpl_->mpi_dims[d] == 1) {
            std::get<0>(m_pimpl_->send_box[2 * d + 0]) = 0;
            std::get<0>(m_pimpl_->send_box[2 * d + 1]) = 0;
            std::get<0>(m_pimpl_->recv_box[2 * d + 0]) = 0;
            std::get<0>(m_pimpl_->recv_box[2 * d + 1]) = 0;
            std::get<1>(m_pimpl_->send_box[2 * d + 0]) = 0;
            std::get<1>(m_pimpl_->send_box[2 * d + 1]) = 0;
            std::get<1>(m_pimpl_->recv_box[2 * d + 0]) = 0;
            std::get<1>(m_pimpl_->recv_box[2 * d + 1]) = 0;
            continue;
        }

        std::get<0>(m_pimpl_->send_box[2 * d + 0])[d] = std::get<0>(m_pimpl_->local_box_)[d];
        std::get<1>(m_pimpl_->send_box[2 * d + 0])[d] = std::get<0>(m_pimpl_->local_box_)[d] + m_pimpl_->m_gw_[d];

        std::get<0>(m_pimpl_->recv_box[2 * d + 0])[d] = std::get<0>(m_pimpl_->local_box_)[d] - m_pimpl_->m_gw_[d];
        std::get<1>(m_pimpl_->recv_box[2 * d + 0])[d] = std::get<0>(m_pimpl_->local_box_)[d];

        std::get<0>(m_pimpl_->send_box[2 * d + 1])[d] = std::get<1>(m_pimpl_->local_box_)[d] - m_pimpl_->m_gw_[d];
        std::get<1>(m_pimpl_->send_box[2 * d + 1])[d] = std::get<1>(m_pimpl_->local_box_)[d];

        std::get<0>(m_pimpl_->recv_box[2 * d + 1])[d] = std::get<1>(m_pimpl_->local_box_)[d];
        std::get<1>(m_pimpl_->recv_box[2 * d + 1])[d] = std::get<1>(m_pimpl_->local_box_)[d] + m_pimpl_->m_gw_[d];

        for (int i = 0; i < d; ++i) {
            std::get<0>(m_pimpl_->send_box[2 * d + 0])[i] = std::get<0>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
            std::get<1>(m_pimpl_->send_box[2 * d + 0])[i] = std::get<0>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];

            std::get<0>(m_pimpl_->recv_box[2 * d + 0])[i] = std::get<0>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
            std::get<1>(m_pimpl_->recv_box[2 * d + 0])[i] = std::get<0>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];

            std::get<0>(m_pimpl_->send_box[2 * d + 1])[i] = std::get<1>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
            std::get<1>(m_pimpl_->send_box[2 * d + 1])[i] = std::get<1>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];

            std::get<0>(m_pimpl_->recv_box[2 * d + 1])[i] = std::get<1>(m_pimpl_->local_box_)[i] - m_pimpl_->m_gw_[i];
            std::get<1>(m_pimpl_->recv_box[2 * d + 1])[i] = std::get<1>(m_pimpl_->local_box_)[i] + m_pimpl_->m_gw_[i];
        }
    }
    GLOBAL_COMM.barrier();
    if (GLOBAL_COMM.rank() == 0) {
        for (int i = 0; i < 6; ++i) { std::cout << m_pimpl_->send_box[i] << std::endl; }
    }
    GLOBAL_COMM.barrier();

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

    for (int i = 0; i < 6; ++i) {
        m_pimpl_->send_size[i] =
            static_cast<int>(std::get<1>(m_pimpl_->send_box[i])[0] - std::get<0>(m_pimpl_->send_box[i])[0]) *
            static_cast<int>(std::get<1>(m_pimpl_->send_box[i])[1] - std::get<0>(m_pimpl_->send_box[i])[1]) *
            static_cast<int>(std::get<1>(m_pimpl_->send_box[i])[2] - std::get<0>(m_pimpl_->send_box[i])[2]);

        m_pimpl_->recv_size[i] =
            static_cast<int>(std::get<1>(m_pimpl_->recv_box[i])[0] - std::get<0>(m_pimpl_->recv_box[i])[0]) *
            static_cast<int>(std::get<1>(m_pimpl_->recv_box[i])[1] - std::get<0>(m_pimpl_->recv_box[i])[1]) *
            static_cast<int>(std::get<1>(m_pimpl_->recv_box[i])[2] - std::get<0>(m_pimpl_->recv_box[i])[2]);

        if (m_pimpl_->send_buffer[i] != nullptr) { delete m_pimpl_->send_buffer[i]; }
        if (m_pimpl_->recv_buffer[i] != nullptr) { delete m_pimpl_->recv_buffer[i]; }

        if (m_pimpl_->send_size[i] > 0) m_pimpl_->send_buffer[i] = operator new(m_pimpl_->send_size[i] * ele_size);
        if (m_pimpl_->recv_size[i] > 0) m_pimpl_->recv_buffer[i] = operator new(m_pimpl_->recv_size[i] * ele_size);
    }
}

void MPIUpdater::TearDown() {
    for (auto *v : m_pimpl_->send_buffer) { delete v; }
    for (auto *v : m_pimpl_->recv_buffer) { delete v; }
    m_pimpl_->m_is_setup_ = false;
}

void MPIUpdater::SetTag(int tag) { m_pimpl_->tag = tag; }

void *MPIUpdater::GetSendBuffer(int i) const { return m_pimpl_->send_buffer[i]; };
index_box_type MPIUpdater::GetSendBufferBox(int i) const { return m_pimpl_->send_box[i]; };

void *MPIUpdater::GetRecvBuffer(int i) const { return m_pimpl_->recv_buffer[i]; };
index_box_type MPIUpdater::GetRecvBufferBox(int i) const { return m_pimpl_->recv_box[i]; };

void MPIUpdater::Update() const {
    if (!isSetUp()) { return; }
    //    CHECK(m_pimpl_->mpi_topology_ndims);

    for (int d = 0; d < m_pimpl_->mpi_topology_ndims; ++d) {
        int left, right;

        MPI_CALL(MPI_Cart_shift(m_pimpl_->comm, d, 1, &left, &right));
        INFORM << "Left : " << left << " Right : " << right << " size : " << m_pimpl_->send_size[d * 2 + 0];
        GLOBAL_COMM.barrier();
        MPI_CALL(MPI_Sendrecv(m_pimpl_->send_buffer[d * 2 + 0], m_pimpl_->send_size[d * 2 + 0], m_pimpl_->ele_type,
                              left, m_pimpl_->tag, m_pimpl_->recv_buffer[d * 2 + 1], m_pimpl_->recv_size[d * 2 + 1],
                              m_pimpl_->ele_type, right, m_pimpl_->tag, m_pimpl_->comm, MPI_STATUS_IGNORE));
        GLOBAL_COMM.barrier();
        MPI_CALL(MPI_Sendrecv(m_pimpl_->send_buffer[d * 2 + 1], m_pimpl_->send_size[d * 2 + 1], m_pimpl_->ele_type,
                              right, m_pimpl_->tag, m_pimpl_->recv_buffer[d * 2 + 0], m_pimpl_->recv_size[d * 2 + 0],
                              m_pimpl_->ele_type, left, m_pimpl_->tag, m_pimpl_->comm, MPI_STATUS_IGNORE));
        GLOBAL_COMM.barrier();
    }
}

}  // namespace parallel {
}  // namespace simpla {