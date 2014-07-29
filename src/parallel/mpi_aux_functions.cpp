/**
 * \file mpi_aux_functions.cpp
 *
 * \date    2014年7月29日  上午8:16:09 
 * \author salmon
 */

#include "mpi_aux_functions.h"

extern "C"
{
#include <mpi.h>
}

#include "message_comm.h"
#include "distributed_array.h"

#include "../utilities/log.h"
#include "../utilities/memory_pool.h"
namespace simpla
{

bool GetMPIType(std::type_index const & t_index, size_t size_in_byte, MPI_Datatype * new_type)
{
	bool is_commited = false;

	if (t_index == std::type_index(typeid(int)))
	{
		*new_type = MPI_INT;
	}
	else if (t_index == std::type_index(typeid(long)))
	{
		*new_type = MPI_LONG;
	}
	else if (t_index == std::type_index(typeid(unsigned long)))
	{
		*new_type = MPI_UNSIGNED_LONG;
	}
	else if (t_index == std::type_index(typeid(float)))
	{
		*new_type = MPI_FLOAT;
	}
	else if (t_index == std::type_index(typeid(double)))
	{
		*new_type = MPI_DOUBLE;
	}
//	else if (t_index == std::type_index(typeid(long double)))
//	{
//		*new_type = MPI_LONG_DOUBLE;
//	}
//	else if (t_index == std::type_index(typeid(std::complex<double>)))
//	{
//		*new_type = MPI_2DOUBLE_COMPLEX;
//	}
//	else if (t_index == std::type_index(typeid(std::complex<float>)))
//	{
//		*new_type = MPI_2COMPLEX;
//	}
	else
	{
		MPI_Type_contiguous(size_in_byte, MPI_BYTE, new_type);
		MPI_Type_commit(new_type);
		is_commited = true;
	}
	return is_commited;
}

bool GetMPIType(DataType const & datatype_desc, MPI_Datatype * new_type)
{
	bool is_commited = false;

	if (datatype_desc.ndims == 0)
	{
		is_commited = GetMPIType(datatype_desc.t_index_, datatype_desc.ele_size_in_byte_, new_type);
	}
	else
	{
		int ndims = datatype_desc.ndims;

		int dims[ndims];

		for (int i = 0; i < ndims; ++i)
		{
			dims[i] = datatype_desc.dimensions_[i];
		}

		MPI_Datatype ele_type;

		GetMPIType(datatype_desc.t_index_, datatype_desc.ele_size_in_byte_, &ele_type);

		MPI_Type_contiguous(ndims, ele_type, new_type);

		MPI_Type_commit(new_type);

		is_commited = true;
	}

	return is_commited;
}

struct MPIDataType
{
	MPI_Datatype type_ = MPI_DATATYPE_NULL;
	bool is_commited_ = false;
	static constexpr unsigned int MAX_NTUPLE_RANK = 10;

	MPIDataType()
	{
	}

	static MPIDataType create(DataType const & datatype);

	template<typename T>
	static MPIDataType create()
	{
		return std::move(create(DataType::create<T>()));
	}

	static MPIDataType create(DataType const & data_type, int NDIMS, int const * outer, int const * inner,
	        int const * start, bool c_order_array = true);

	~MPIDataType()
	{
		if (is_commited_)
			MPI_Type_free(&type_);
	}

	MPI_Datatype const & type(...) const
	{
		return type_;
	}

};
MPIDataType MPIDataType::create(DataType const & datatype)
{
	MPIDataType res;
	res.is_commited_ = (GetMPIType(datatype, &res.type_));
	return res;

}
MPIDataType MPIDataType::create(DataType const & datatype, int NDIMS, int const * outer, int const * inner,
        int const * start, bool c_order_array)
{
	MPIDataType res;

	const int v_ndims = datatype.ndims;

	int outer1[NDIMS + v_ndims];
	int inner1[NDIMS + v_ndims];
	int start1[NDIMS + v_ndims];
	for (int i = 0; i < NDIMS; ++i)
	{
		outer1[i] = outer[i];
		inner1[i] = inner[i];
		start1[i] = start[i];
	}

	for (int i = 0; i < v_ndims; ++i)
	{
		outer1[NDIMS + i] = datatype.dimensions_[i];
		inner1[NDIMS + i] = datatype.dimensions_[i];
		start1[NDIMS + i] = 0;
	}

	for (int i = 0; i < v_ndims; ++i)
	{
		start1[NDIMS + i] = 0;
	}
	MPI_Datatype ele_type;

	GetMPIType(datatype.t_index_, datatype.ele_size_in_byte_, &ele_type);

	MPI_Type_create_subarray(NDIMS + v_ndims, outer1, inner1, start1, (c_order_array ? MPI_ORDER_C : MPI_ORDER_FORTRAN),
	        ele_type, &res.type_);

	MPI_Type_commit(&res.type_);

	res.is_commited_ = true;

	return res;
}

/**
 * @param pos in {0,count} out {begin,shape}
 */
template<typename Integral>
std::tuple<Integral, Integral> sync_global_location(Integral count)
{
	Integral begin = 0;

	if ( GLOBAL_COMM.is_ready() && GLOBAL_COMM.get_size() > 1)
	{

		auto communicator = GLOBAL_COMM.comm();

		int num_of_process = GLOBAL_COMM.get_size();
		int porcess_number = GLOBAL_COMM.get_rank();

		MPIDataType m_type =MPIDataType::create<Integral>();

		std::vector<Integral> buffer;

		if (porcess_number == 0)
		buffer.resize(num_of_process);

		MPI_Gather(&count, 1, m_type.type(), &buffer[0], 1, m_type.type(), 0, communicator);

		MPI_Barrier(communicator);

		if (porcess_number == 0)
		{
			for (int i = 1; i < num_of_process; ++i)
			{
				buffer[i] += buffer[i - 1];
			}
			buffer[0] = count;
			count = buffer[num_of_process - 1];

			for (int i = num_of_process - 1; i > 0; --i)
			{
				buffer[i] = buffer[i - 1];
			}
			buffer[0] = 0;
		}
		MPI_Barrier(communicator);
		MPI_Scatter(&buffer[0], 1, m_type.type(), &begin, 1, m_type.type(), 0, communicator);
		MPI_Bcast(&count, 1, m_type.type(), 0, communicator);
	}

	return std::make_tuple(begin, count);

}
inline MPI_Op get_MPI_Op(std::string const & op_c)
{
	MPI_Op op = MPI_SUM;

	if (op_c == "Max")
	{
		op = MPI_MAX;
	}
	else if (op_c == "Min")
	{
		op = MPI_MIN;
	}
	else if (op_c == "Sum")
	{
		op = MPI_SUM;
	}
	else if (op_c == "Prod")
	{
		op = MPI_PROD;
	}
	else if (op_c == "LAND")
	{
		op = MPI_LAND;
	}
	else if (op_c == "LOR")
	{
		op = MPI_LOR;
	}
	else if (op_c == "BAND")
	{
		op = MPI_BAND;
	}
	else if (op_c == "Sum")
	{
		op = MPI_BOR;
	}
	else if (op_c == "Sum")
	{
		op = MPI_MAXLOC;
	}
	else if (op_c == "Sum")
	{
		op = MPI_MINLOC;
	}
	return op;
}

void reduce(void const* send_data, void * recv_data, size_t count, DataType const & data_type, std::string const & op_c)
{
	auto m_type = MPIDataType::create(data_type);

	auto communicator = GLOBAL_COMM.comm();
	GLOBAL_COMM.barrier();
	MPI_Reduce(const_cast<void*>(send_data), (recv_data), count, m_type.type(), get_MPI_Op(op_c), 0, communicator);
	GLOBAL_COMM.barrier();

}

void allreduce(void const* send_data, void * recv_data, size_t count, DataType const & data_type,
        std::string const & op_c)
{

	auto m_type = MPIDataType::create(data_type);

	auto communicator = GLOBAL_COMM.comm();
	GLOBAL_COMM.barrier();
	MPI_Allreduce(const_cast<void*>(send_data), reinterpret_cast<void*>(recv_data), count, m_type.type(),
	        get_MPI_Op(op_c), communicator);
	GLOBAL_COMM.barrier();

}



std::tuple<std::shared_ptr<ByteType>, int> update_ghost_unorder(void const* send_buffer, std::vector<

std::tuple<int, // dest;
        int, // send_tag;
        int, // recv_tag;
        int, // send buffer begin;
        int  // send buffer size;
        >> const & info)
{
	GLOBAL_COMM.barrier();

	MPI_Request requests[info.size() * 2];

	int req_count = 0;

	// send
	for (auto const & item : info)
	{

		MPI_Isend( reinterpret_cast<ByteType*>(const_cast<void* >(send_buffer))+std::get<3>(item) ,
				std::get<4>(item), MPI_BYTE, std::get<0>(item), std::get<1>(item),
				GLOBAL_COMM.comm(), &requests[req_count]);

		++req_count;
	}

	std::vector<int> mem_size;

	for (auto const & item : info)
	{
		MPI_Status status;

		MPI_Probe(std::get<0>(item), std::get<2>(item), GLOBAL_COMM.comm(), &status);

		// When probe returns, the status object has the size and other
		// attributes of the incoming message. Get the size of the message
		int tmp = 0;
		MPI_Get_count(&status, MPI_BYTE, &tmp);

		if (tmp == MPI_UNDEFINED)
		{
			RUNTIME_ERROR("Update Ghosts Particle fail");
		}
		else
		{
			mem_size.push_back(tmp);
		}

	}
	int recv_buffer_size=std::accumulate(mem_size.begin(),mem_size.end(),0);
	auto recv_buffer = MEMPOOL.allocate_byte_shared_ptr(recv_buffer_size);

	int pos = 0;
	for (int i = 0; i < info.size(); ++i)
	{

		MPI_Irecv(recv_buffer.get() + pos, mem_size[i], MPI_BYTE, std::get<0>(info[i]), std::get<2>(info[i]),
				GLOBAL_COMM.comm(), &requests[req_count] );

		pos+= mem_size[i];
		++req_count;
	}

	MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
	GLOBAL_COMM.barrier();

	return std::make_tuple(recv_buffer,recv_buffer_size);
}
}
		// namespace simpla
