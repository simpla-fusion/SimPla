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

	std::string init(int argc = 0, char** argv = nullptr);

	void close();

	MPI_Comm comm();
	MPI_Info info();

	void barrier();

	bool is_valid() const;

	int process_num() const;

	int num_of_process() const;

	int generate_object_id();

//	void set_num_of_threads(int num);
//
//	unsigned int get_num_of_threads() const;

	nTuple<int, 3> topology() const;

	void topology(int nx, int ny = 1, int nz = 1);

	int get_neighbour(int disp_i, int disp_j = 0, int disp_k = 0) const;

	template<typename TI>
	int get_neighbour(TI const &d) const
	{
		return get_neighbour(d[0], d[1], d[2]);
	}

	nTuple<int, 3> get_coordinate(int rank) const;

	void decompose(int ndims, size_t * count, size_t * offset) const;
	template<typename TI>
	void decompose(int ndims, TI * count, TI * offset) const
	{
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> t_count, t_offset;
		t_count = count;
		t_offset = offset;
		decompose(ndims, &t_count[0], &t_offset[0]);
		for (int i = 0; i < ndims; ++i)
		{
			count[i] = t_count[i];
			offset[i] = t_offset[i];
		}
	}
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

}
// namespace simpla

#endif /* MPI_COMM_H_ */
