/**
 * @file MPIComm.h
 *
 *  created on: 2014-5-12
 *      Author: salmon
 */

#ifndef MPI_COMM_H_
#define MPI_COMM_H_

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/utilities/SingletonHolder.h>
#include <algorithm>
#include <cstdbool>
#include <cstddef>
#include <memory>
#include <typeindex>

namespace simpla {
namespace parallel {

class MPIComm {
   public:
    MPIComm();
    MPIComm(int argc, char **argv);
    ~MPIComm();
    void Initialize(int argc = 0, char **argv = nullptr);
    void Finalize();
    void barrier();
    bool is_valid() const;
    int process_num() const;
    int num_of_process() const;
    size_type generate_object_id();
    int topology(int *mpi_topo_ndims, int *mpi_topo_dims, int *periods, int *mpi_topo_coord) const;
    void CartShift(int dirction, int disp, int *left, int *right) const;
    int rank() const;
    int size() const;
    int get_rank(int const *d) const;

    void SendRecv(const void *sendbuf, int sendcount, size_type sendtype_hash, int dest, int sendtag, void *recvbuf,
                  int recvcount, size_type recvtype_hash, int source, int recvtag);
    //    std::tuple<int, int, int> make_send_recv_tag(size_t prefix, const nTuple<int, 3> &m_global_start_);
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

std::string bcast_string(std::string const &str = "", int root = 0);

std::string gather_string(std::string const &str = "", int root = 0, size_type *num = nullptr,
                          size_type **disp = nullptr);

#define GLOBAL_COMM SingletonHolder<::simpla::parallel::MPIComm>::instance()
}  // namespace parallel{
}  // namespace simpla{

#endif /* MPI_COMM_H_ */
