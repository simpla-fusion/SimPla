/**
 * @file MPIComm.h
 *
 *  created on: 2014-5-12
 *      Author: salmon
 */

#ifndef MPI_COMM_H_
#define MPI_COMM_H_

#include <stddef.h>
#include <cstdbool>
#include <memory>
#include <algorithm>


#include <mpi.h>


#include "../gtl/nTuple.h"
#include "../gtl/design_pattern/SingletonHolder.h"
//#include "../gtl/Utilities.h"

namespace simpla { namespace parallel
{

class MPIComm
{

public:
    static constexpr int NDIMS = 3;

    MPIComm();

    MPIComm(int argc, char **argv);

    ~MPIComm();

    void init(int argc = 0, char **argv = nullptr);

    void close();

    static std::string help_message();

    MPI_Comm comm();

    MPI_Info info();

    void barrier();

    bool is_valid() const;

    int process_num() const;

    int num_of_process() const;

    int generate_object_id();


    nTuple<int, 3> topology() const;

    void topology(nTuple<int, 3> const &d);

    int get_neighbour(nTuple<int, 3> const &d) const;

    nTuple<int, 3> coordinate(int rank = -1) const;

    int get_rank() const;

    int get_rank(nTuple<int, 3> const &d) const;

    std::tuple<int, int, int> make_send_recv_tag(size_t prefix, const nTuple<int, 3> &offset);

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> pimpl_;
};

#define GLOBAL_COMM   SingletonHolder<::simpla::parallel::MPIComm>::instance()

#define MPI_ERROR(_CMD_)                                           \
{                                                                  \
    int _mpi_error_code_ = _CMD_;                                  \
    if (_mpi_error_code_ != MPI_SUCCESS)                           \
    {                                                              \
        char _error_msg[MPI_MAX_ERROR_STRING];                     \
        MPI_Error_string(_mpi_error_code_, _error_msg, nullptr);   \
         THROW_EXCEPTION_RUNTIME_ERROR(_error_msg);                \
    }                                                              \
}
}}//namespace simpla{namespace parallel{


#endif /* MPI_COMM_H_ */
