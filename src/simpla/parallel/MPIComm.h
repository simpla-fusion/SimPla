/**
 * @file MPIComm.h
 *
 *  created on: 2014-5-12
 *      Author: salmon
 */

#ifndef MPI_COMM_H_
#define MPI_COMM_H_

#include <mpi.h>
#include <algorithm>
#include <cstdbool>
#include <cstddef>
#include <memory>
#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/utilities/SingletonHolder.h"

namespace simpla {
namespace parallel {

class MPIComm {
   public:
    MPIComm();
    MPIComm(int argc, char **argv);
    ~MPIComm();
    void Initialize(int argc = 0, char **argv = nullptr);
    void Finalize();
    MPI_Comm comm() const;
    //    MPI_Info info();
    void barrier();
    bool is_valid() const;
    int process_num() const;
    int num_of_process() const;
    size_type generate_object_id();
    int topology(int *mpi_topo_ndims, int *mpi_topo_dims, int *periods, int *mpi_topo_coord) const;
    int rank() const;
    int size() const;
    int get_rank(int const *d) const;

    //    std::tuple<int, int, int> make_send_recv_tag(size_t prefix, const nTuple<int, 3> &m_global_start_);

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

std::string bcast_string(std::string const &str = "");

#define GLOBAL_COMM SingletonHolder<::simpla::parallel::MPIComm>::instance()

#define MPI_CALL(_CMD_)                                              \
    {                                                                \
        int _mpi_error_code_ = _CMD_;                                \
        if (_mpi_error_code_ != MPI_SUCCESS) {                       \
            char _error_msg[MPI_MAX_ERROR_STRING];                   \
            MPI_Error_string(_mpi_error_code_, _error_msg, nullptr); \
            THROW_EXCEPTION_RUNTIME_ERROR(_error_msg);               \
        }                                                            \
    }
}
}  // namespace simpla{namespace parallel{

#endif /* MPI_COMM_H_ */
