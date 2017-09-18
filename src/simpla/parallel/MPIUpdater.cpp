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

    index_box_type local_box_;
    index_box_type send_box[6];
    index_box_type recv_box[6];

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
void MPIUpdater::SetIndexBox(index_box_type const &idx_box) { m_pimpl_->local_box_ = idx_box; }
index_box_type MPIUpdater::GetIndexBox() const { return m_pimpl_->local_box_; }

bool MPIUpdater::isSetUp() const { return m_pimpl_->m_is_setup_; }

void MPIUpdater::SetUp() {
    m_pimpl_->m_is_setup_ = true;
    int topo_type = MPI_CART;

    MPI_CALL(MPI_Topo_test(m_pimpl_->comm, &topo_type));

    assert(topo_type == MPI_CART);

    int mpi_topology_ndims = 0;

    MPI_CALL(MPI_Cartdim_get(m_pimpl_->comm, &mpi_topology_ndims));

    assert(mpi_topology_ndims <= m_pimpl_->ndims);

    //    for (int i = 0; i < m_pimpl_->ndims; ++i) { m_pimpl_->dims[i] = shape[i]; }
    //
    //    m_pimpl_->num_of_neighbour = mpi_topology_ndims * 2;
    //
    //    for (int d = 0; d < mpi_topology_ndims; ++d) {
    //        m_pimpl_->send_displs[2 * d + 0] = 0;
    //        m_pimpl_->send_displs[2 * d + 1] = 0;
    //        m_pimpl_->recv_displs[2 * d + 0] = 0;
    //        m_pimpl_->recv_displs[2 * d + 1] = 0;
    //
    //        if (m_pimpl_->dims[d] == 1) {
    //            m_pimpl_->send_count[2 * d + 0] = 0;
    //            m_pimpl_->send_count[2 * d + 1] = 0;
    //            m_pimpl_->recv_count[2 * d + 0] = 0;
    //            m_pimpl_->recv_count[2 * d + 1] = 0;
    //
    //            for (int i = 0; i < ndims; ++i) {
    //                m_pimpl_->s_count[2 * d + 0][i] = 0;
    //                m_pimpl_->s_start[2 * d + 0][i] = 0;
    //                m_pimpl_->s_count[2 * d + 1][i] = 0;
    //                m_pimpl_->s_start[2 * d + 1][i] = 0;
    //                m_pimpl_->r_count[2 * d + 0][i] = 0;
    //                m_pimpl_->r_start[2 * d + 0][i] = 0;
    //                m_pimpl_->r_count[2 * d + 1][i] = 0;
    //                m_pimpl_->r_start[2 * d + 1][i] = 0;
    //            }
    //
    //        } else {
    //            m_pimpl_->send_count[2 * d + 0] = 1;
    //            m_pimpl_->send_count[2 * d + 1] = 1;
    //            m_pimpl_->recv_count[2 * d + 0] = 1;
    //            m_pimpl_->recv_count[2 * d + 1] = 1;
    //
    //            for (int i = 0; i < ndims; ++i) {
    //                if (i >= mpi_sync_start_dims && i < d + mpi_sync_start_dims) {
    //                    m_pimpl_->s_count[2 * d + 0][i] = m_pimpl_->dims[i];
    //                    m_pimpl_->s_start[2 * d + 0][i] = 0;
    //                    m_pimpl_->s_count[2 * d + 1][i] = m_pimpl_->dims[i];
    //                    m_pimpl_->s_start[2 * d + 1][i] = 0;
    //
    //                    m_pimpl_->r_count[2 * d + 0][i] = m_pimpl_->dims[i];
    //                    m_pimpl_->r_start[2 * d + 0][i] = 0;
    //                    m_pimpl_->r_count[2 * d + 1][i] = m_pimpl_->dims[i];
    //                    m_pimpl_->r_start[2 * d + 1][i] = 0;
    //                } else if (i == d + mpi_sync_start_dims) {
    //                    m_pimpl_->s_count[2 * d + 0][i] = start[i];
    //                    m_pimpl_->s_start[2 * d + 0][i] = start[i];
    //                    m_pimpl_->s_count[2 * d + 1][i] = (m_pimpl_->dims[i] - count[i] - start[i]);
    //                    m_pimpl_->s_start[2 * d + 1][i] = (start[i] + count[i] - m_pimpl_->s_count[2 * d + 1][i]);
    //
    //                    m_pimpl_->r_count[2 * d + 0][i] = start[i];
    //                    m_pimpl_->r_start[2 * d + 0][i] = 0;
    //                    m_pimpl_->r_count[2 * d + 1][i] = (m_pimpl_->dims[i] - count[i] - start[i]);
    //                    m_pimpl_->r_start[2 * d + 1][i] = m_pimpl_->dims[i] - m_pimpl_->s_count[2 * d + 1][i];
    //                } else {
    //                    m_pimpl_->s_count[2 * d + 0][i] = count[i];
    //                    m_pimpl_->s_start[2 * d + 0][i] = start[i];
    //                    m_pimpl_->s_count[2 * d + 1][i] = count[i];
    //                    m_pimpl_->s_start[2 * d + 1][i] = start[i];
    //
    //                    m_pimpl_->r_count[2 * d + 0][i] = count[i];
    //                    m_pimpl_->r_start[2 * d + 0][i] = start[i];
    //                    m_pimpl_->r_count[2 * d + 1][i] = count[i];
    //                    m_pimpl_->r_start[2 * d + 1][i] = start[i];
    //                };
    //                m_pimpl_->send_count[2 * d + 0] *= m_pimpl_->s_count[2 * d + 0][i];
    //                m_pimpl_->send_count[2 * d + 1] *= m_pimpl_->s_count[2 * d + 1][i];
    //                m_pimpl_->recv_count[2 * d + 0] *= m_pimpl_->r_count[2 * d + 0][i];
    //                m_pimpl_->recv_count[2 * d + 1] *= m_pimpl_->r_count[2 * d + 1][i];
    //            }
    //        }
    //    }
    //
    //    m_pimpl_->strides[ndims - 1] = 1;
    //
    //    for (int i = ndims - 2; i >= 0; --i) { m_pimpl_->strides[i] = m_pimpl_->dims[i + 1] * m_pimpl_->strides[i +
    //    1]; }

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
        size_type send_size =
            static_cast<size_type>(std::get<1>(m_pimpl_->send_box[i])[0] - std::get<0>(m_pimpl_->send_box[i])[0]) *
            static_cast<size_type>(std::get<1>(m_pimpl_->send_box[i])[1] - std::get<0>(m_pimpl_->send_box[i])[1]) *
            static_cast<size_type>(std::get<1>(m_pimpl_->send_box[i])[2] - std::get<0>(m_pimpl_->send_box[i])[2]);

        size_type recv_size =
            static_cast<size_type>(std::get<1>(m_pimpl_->recv_box[i])[0] - std::get<0>(m_pimpl_->recv_box[i])[0]) *
            static_cast<size_type>(std::get<1>(m_pimpl_->recv_box[i])[1] - std::get<0>(m_pimpl_->recv_box[i])[1]) *
            static_cast<size_type>(std::get<1>(m_pimpl_->recv_box[i])[2] - std::get<0>(m_pimpl_->recv_box[i])[2]);

        if (m_pimpl_->send_buffer[i] != nullptr) { delete m_pimpl_->send_buffer[i]; }
        if (m_pimpl_->recv_buffer[i] != nullptr) { delete m_pimpl_->recv_buffer[i]; }

        m_pimpl_->send_buffer[i] = operator new(send_size *ele_size);
        m_pimpl_->recv_buffer[i] = operator new(recv_size *ele_size);
    }
}

void MPIUpdater::TearDown() {
    for (auto *v : m_pimpl_->send_buffer) { delete v; }
    for (auto *v : m_pimpl_->recv_buffer) { delete v; }
    m_pimpl_->m_is_setup_ = false;
}

void MPIUpdater::SetTag(int tag) { m_pimpl_->tag = tag; }

std::tuple<void *, index_box_type> MPIUpdater::GetSendBuffer(int i) const {
    return std::make_tuple(m_pimpl_->send_buffer[i], m_pimpl_->send_box[i]);
};
std::tuple<void *, index_box_type> MPIUpdater::GetRecvBuffer(int i) const {
    return std::make_tuple(m_pimpl_->recv_buffer[i], m_pimpl_->recv_box[i]);
};
void MPIUpdater::Update() const {
    for (int d = 0; d < m_pimpl_->ndims; ++d) {
        int left, right;

        MPI_CALL(MPI_Cart_shift(m_pimpl_->comm, d, 1, &left, &right));

        MPI_CALL(MPI_Sendrecv(m_pimpl_->send_buffer[d * 2 + 0], m_pimpl_->send_size[d * 2 + 0], m_pimpl_->ele_type,
                              left, m_pimpl_->tag, m_pimpl_->recv_buffer[d * 2 + 1], m_pimpl_->recv_size[d * 2 + 1],
                              m_pimpl_->ele_type, right, m_pimpl_->tag, m_pimpl_->comm, MPI_STATUS_IGNORE));

        MPI_CALL(MPI_Sendrecv(m_pimpl_->send_buffer[d * 2 + 1], m_pimpl_->send_size[d * 2 + 1], m_pimpl_->ele_type,
                              right, m_pimpl_->tag, m_pimpl_->recv_buffer[d * 2 + 0], m_pimpl_->recv_size[d * 2 + 0],
                              m_pimpl_->ele_type, left, m_pimpl_->tag, m_pimpl_->comm, MPI_STATUS_IGNORE));
    }
}

}  // namespace parallel {
}  // namespace simpla {