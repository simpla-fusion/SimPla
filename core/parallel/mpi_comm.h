/*
 * mpi_comm.h
 *
 *  created on: 2014-5-12
 *      Author: salmon
 */

#ifndef MPI_COMM_H_
#define MPI_COMM_H_

#include <stddef.h>
#include <cstdbool>
#include <memory>

#include "../gtl/ntuple.h"

extern "C"
{
#include <mpi.h>
}
#include <algorithm>

#include "../dataset/dataset.h"
#include "../gtl/design_pattern/singleton_holder.h"
#include "../utilities/utilities.h"

namespace simpla
{

class MPIComm
{

public:

	MPIComm();

	MPIComm(int argc, char** argv);

	~MPIComm();

	void init(int argc = 0, char** argv = nullptr);

	void close();

	MPI_Comm comm();
	MPI_Info info();

	bool is_valid() const;

	int process_num() const;

	int num_of_process() const;

//	void set_num_of_threads(int num);
//
//	unsigned int get_num_of_threads() const;

	nTuple<int, 3> const & topology() const;

	void topology(int nx, int ny = 1, int nz = 1);

	int get_neighbour(int disp_i, int disp_j = 0, int disp_k = 0) const;

	template<typename TI>
	int get_neighbour(TI const &d) const
	{
		return get_neighbour(d[0], d[1], d[2]);
	}

	nTuple<int, 3> get_coordinate(int rank) const;

	void decompose(int ndims, size_t * offset, size_t * count) const;

private:
	struct pimpl_s;
	std::unique_ptr<pimpl_s> pimpl_;
}
;
#define GLOBAL_COMM   SingletonHolder<simpla::MPIComm>::instance()

#define MPI_ERROR(_CMD_)                                                   \
{                                                                          \
	int _mpi_error_code_ = MPI_SUCCESS;                                    \
	if ( ( _mpi_error_code_=  _CMD_) != MPI_SUCCESS)                       \
	{                                                                      \
		char _error_msg[MPI_MAX_ERROR_STRING];                             \
		MPI_Error_string(_mpi_error_code_, _error_msg, nullptr);           \
		RUNTIME_ERROR(_error_msg);                                         \
	}                                                                      \
}

struct mpi_ghosts_shape_s
{
	nTuple<int, 3> coord_shift;

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_offset;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_offset;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count;
};

void get_ghost_shape(size_t ndims, size_t const * dims, size_t const * offset,
		size_t const * stride, size_t const * count, size_t const * block,
		size_t const * ghost_width,
		std::vector<mpi_ghosts_shape_s>* send_recv_list)  ;
}
// namespace simpla

#endif /* MPI_COMM_H_ */
