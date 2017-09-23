/**
 * @file mpi_comm.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */
#include "MPIComm.h"
#include <mpi.h>
#include <simpla/algebra/nTuple.ext.h>
#include <simpla/utilities/parse_command_line.h>
#include <simpla/utilities/type_cast.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "simpla/SIMPLA_config.h"
#include "simpla/utilities/Log.h"
namespace simpla {
namespace parallel {

struct MPIComm::pimpl_s {
    static constexpr int MAX_NUM_OF_DIMS = 3;
    MPI_Comm m_comm_ = MPI_COMM_NULL;
    size_type m_object_id_count_ = 0;
    int m_topology_ndims_ = 3;
    int m_topology_dims_[3] = {0, 1, 1};
};

MPIComm::MPIComm() : m_pimpl_(new pimpl_s) {}

MPIComm::MPIComm(int argc, char **argv) : MPIComm() { Initialize(argc, argv); }

MPIComm::~MPIComm() { Finalize(); }

int MPIComm::process_num() const { return rank(); }

int MPIComm::num_of_process() const { return size(); }

int MPIComm::rank() const {
    int res = 0;
    if (m_pimpl_->m_comm_ != MPI_COMM_NULL) { MPI_Comm_rank(m_pimpl_->m_comm_, &res); }
    return res;
}

int MPIComm::size() const {
    int res = 1;
    if (m_pimpl_->m_comm_ != MPI_COMM_NULL) { MPI_Comm_size(m_pimpl_->m_comm_, &res); }
    return res;
}

int MPIComm::get_rank(int const *d) const {
    int res = 0;
    MPI_CALL(MPI_Cart_rank(m_pimpl_->m_comm_, (int *)d, &res));
    return res;
}

void MPIComm::Initialize(int argc, char **argv) {
    MPI_CALL(MPI_Init(&argc, &argv));
    parse_cmd_line(argc, argv, [&](std::string const &opt, std::string const &value) -> int {
        if (opt == "mpi_topology") {
            auto v = type_cast<nTuple<int, 3> >(value);
            m_pimpl_->m_topology_dims_[0] = v[0];
            m_pimpl_->m_topology_dims_[1] = v[1];
            m_pimpl_->m_topology_dims_[2] = v[2];
        }
        return CONTINUE;
    });

    m_pimpl_->m_object_id_count_ = 0;
    int m_num_process_;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &m_num_process_));
    if (m_num_process_ > 1) {
        int m_topology_coord_[3] = {0, 0, 0};
        MPI_CALL(MPI_Dims_create(m_num_process_, m_pimpl_->m_topology_ndims_, m_pimpl_->m_topology_dims_));
        int periods[m_pimpl_->m_topology_ndims_];
        for (int i = 0; i < m_pimpl_->m_topology_ndims_; ++i) { periods[i] = true; }
        MPI_CALL(MPI_Cart_create(MPI_COMM_WORLD, m_pimpl_->m_topology_ndims_, m_pimpl_->m_topology_dims_, periods,
                                 MPI_ORDER_C, &m_pimpl_->m_comm_));
        logger::set_mpi_comm(rank(), size());

        MPI_CALL(MPI_Cart_coords(m_pimpl_->m_comm_, rank(), m_pimpl_->m_topology_ndims_, m_topology_coord_));

        INFORM << "MPI communicator is initialized! "
                  "[("
               << m_topology_coord_[0] << "," << m_topology_coord_[1] << "," << m_topology_coord_[2] << ")/("
               << m_pimpl_->m_topology_dims_[0] << "," << m_pimpl_->m_topology_dims_[1] << ","
               << m_pimpl_->m_topology_dims_[2] << ")/(" << periods[0] << "," << periods[1] << "," << periods[2] << "]"
               << std::endl;
    }
}

size_type MPIComm::generate_object_id() {
    assert(m_pimpl_ != nullptr);

    ++(m_pimpl_->m_object_id_count_);

    return m_pimpl_->m_object_id_count_;
}

MPI_Comm MPIComm::comm() const { return m_pimpl_->m_comm_; }
//
// MPI_Info MPIComm::info() {
//    assert(m_pack_ != nullptr);
//    return MPI_INFO_NULL;
//}

void MPIComm::barrier() {
    if (m_pimpl_->m_comm_ != MPI_COMM_NULL) { MPI_Barrier(m_pimpl_->m_comm_); }
}

bool MPIComm::is_valid() const { return ((!!m_pimpl_) && m_pimpl_->m_comm_ != MPI_COMM_NULL) && num_of_process() > 1; }

int MPIComm::topology(int *mpi_topo_ndims, int *mpi_topo_dims, int *periods, int *mpi_topo_coord) const {
    *mpi_topo_ndims = 0;
    if (mpi_topo_dims == nullptr || periods == nullptr || mpi_topo_coord == nullptr) { return SP_SUCCESS; }
    if (m_pimpl_->m_comm_ == MPI_COMM_NULL) {
        *mpi_topo_dims = 1;
        *periods = 1;
        *mpi_topo_coord = 0;

    } else {
        int tope_type = MPI_CART;

        MPI_CALL(MPI_Topo_test(m_pimpl_->m_comm_, &tope_type));

        if (tope_type == MPI_CART) {
            MPI_CALL(MPI_Cartdim_get(m_pimpl_->m_comm_, mpi_topo_ndims));

            MPI_CALL(MPI_Cart_get(m_pimpl_->m_comm_, *mpi_topo_ndims, mpi_topo_dims, periods, mpi_topo_coord));
        }
    }
    return SP_SUCCESS;
};

void MPIComm::Finalize() {
    if (m_pimpl_ != nullptr && m_pimpl_->m_comm_ != MPI_COMM_NULL) {
        VERBOSE << "MPI Communicator is closed!" << std::endl;

        MPI_CALL(MPI_Finalize());

        m_pimpl_->m_comm_ = MPI_COMM_NULL;
    }
}

std::string bcast_string(std::string const &str, int root) {
    if (GLOBAL_COMM.size() <= 1) { return str; };
    std::string s_buffer = str;
    int name_len;
    if (GLOBAL_COMM.process_num() == root) { name_len = s_buffer.size(); }
    MPI_Bcast(&name_len, 1, MPI_INT, 0, GLOBAL_COMM.m_pimpl_->m_comm_);
    std::vector<char> buffer(static_cast<size_type>(name_len));
    if (GLOBAL_COMM.process_num() == root) { std::copy(s_buffer.begin(), s_buffer.end(), buffer.begin()); }
    MPI_Bcast((&buffer[0]), name_len, MPI_CHAR, 0, GLOBAL_COMM.m_pimpl_->m_comm_);
    buffer.push_back('\0');
    if (GLOBAL_COMM.process_num() != root) { s_buffer = &buffer[0]; }
    return s_buffer;
}

std::string gather_string(std::string const &str, int root, size_type *num, size_type **disp) {
    if (GLOBAL_COMM.size() <= 1) { return str; }
    bool do_bcast = false;
    if (root < 0) {
        root = 0;
        do_bcast = true;
    }

    std::string res;

    if (GLOBAL_COMM.rank() == root) {
        std::string grid_list;

        /*
         * Now, we Gather the string lengths to the root process,
         * so we can create the buffer into which we'll receive the strings
         */

        int *recvcounts = nullptr;

        /* Only root has the received data */
        if (GLOBAL_COMM.rank() == root) recvcounts = reinterpret_cast<int *>(malloc(GLOBAL_COMM.size() * sizeof(int)));

        int str_len = static_cast<int>(str.size());

        GLOBAL_COMM.barrier();
        MPI_Gather(&str_len, 1, MPI_INT, recvcounts, 1, MPI_INT, root, MPI_COMM_WORLD);
        GLOBAL_COMM.barrier();

        /*
         * Figure out the total length of string,
         * and displacements for each rank
         */

        int totlen = 0;
        int *displs = nullptr;
        char *recvbuf = nullptr;

        displs = reinterpret_cast<int *>(malloc(GLOBAL_COMM.size() * sizeof(int)));

        displs[0] = 0;
        totlen += recvcounts[0] + 1;

        for (int i = 1; i < GLOBAL_COMM.size(); i++) {
            totlen += recvcounts[i] + 1; /* plus one for space or \0 after words */
            displs[i] = displs[i - 1] + recvcounts[i - 1] + 1;
        }

        /* allocate string, pre-fill with spaces and null terminator */
        recvbuf = reinterpret_cast<char *>(malloc(totlen * sizeof(char)));
        for (int i = 0; i < totlen - 1; i++) recvbuf[i] = ' ';
        recvbuf[totlen - 1] = '\0';

        /*
         * Now we have the receive buffer, counts, and displacements, and
         * can gather the strings
         */
        MPI_Gatherv(str.c_str(), static_cast<int>(str.size()), MPI_CHAR, recvbuf, recvcounts, displs, MPI_CHAR, root,
                    MPI_COMM_WORLD);
        GLOBAL_COMM.barrier();
        recvbuf[totlen - 1] = '\0';
        res = recvbuf;
        free(recvbuf);
        free(displs);
    } else {
        int str_len = static_cast<int>(str.size());
        GLOBAL_COMM.barrier();
        MPI_Gather(&str_len, 1, MPI_INT, nullptr, 1, MPI_INT, root, MPI_COMM_WORLD);
        GLOBAL_COMM.barrier();

        MPI_Gatherv(str.c_str(), static_cast<int>(str.size()), MPI_CHAR, nullptr, nullptr, nullptr, MPI_CHAR, root,
                    MPI_COMM_WORLD);
        GLOBAL_COMM.barrier();
    }
    if (do_bcast) { res = bcast_string(res, root); }
    return res;
}
}
}  // namespace simpla{namespace parallel{
