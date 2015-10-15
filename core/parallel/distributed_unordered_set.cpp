/**
 * @file distributed_object.cpp
 * @author salmon
 * @date 2015-10-15.
 */
#include "../gtl/utilities/log.h"
#include "distributed_unordered_set.h"
#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_update.h"

namespace simpla
{
//! Default constructor
DistributedObject::DistributedObject()
{
	m_object_id_ =
			SingletonHolder<simpla::MPIComm>::instance().generate_object_id();
}

DistributedObject::DistributedObject(const DistributedObject &)
{
	m_object_id_ =
			SingletonHolder<simpla::MPIComm>::instance().generate_object_id();
}

//! destroy.
DistributedObject::~DistributedObject()
{
}

bool DistributedObject::is_ready() const
{
	//FIXME this is not multi-threads safe

	if (is_valid())
	{
		return false;
	}
	else if (m_mpi_requests_.size() > 0)
	{
		int flag = 0;
		MPI_ERROR(MPI_Testall(static_cast<int>(m_mpi_requests_.size()), //
				const_cast<MPI_Request *>(&m_mpi_requests_[0]),//
				&flag, MPI_STATUSES_IGNORE));

		return flag != 0;
	}

	return true;

}

void DistributedObject::deploy(std::vector<dist_sync_connection> const &ghost_shape)
{
	int ndims = 3;

	auto ds = dataset();

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> l_dims;

	auto d_shape = ds.dataspace.local_shape();

	ndims = d_shape.ndims;

	l_dims = d_shape.dimensions;

	make_send_recv_list(object_id(), ds.datatype, ndims, &l_dims[0],
			ghost_shape, &m_send_recv_list_);
}

void DistributedObject::sync()
{
	if (m_send_recv_list_.size() > 0)
	{
		sync_update_continue(m_send_recv_list_, dataset().data.get());
	}
}

void DistributedObject::async()
{
	if (m_send_recv_list_.size() > 0)
	{
		sync_update_continue(m_send_recv_list_, dataset().data.get(), &(m_mpi_requests_));
	}
}


void DistributedObject::wait() const
{
	if (!is_valid())
	{
		ERROR("Object is not depolied!");
	}

	wait_all_request(const_cast<std::vector<MPI_Request> *>(&m_mpi_requests_));
}


struct dist_sync_connection
{
	nTuple<int, 3> coord_shift;

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_offset;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_offset;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count;
};

void make_dist_connection(int ndims, size_t const *offset, size_t const *stride,
		size_t const *count, size_t const *block, size_t const *ghost_width,
		std::vector<dist_sync_connection> *dist_connect);

void get_ghost_shape(int ndims, size_t const *l_offset,
		size_t const *l_stride, size_t const *l_count, size_t const *l_block,
		size_t const *ghost_width,
		std::vector<dist_sync_connection> *dist_connect)
{
	dist_connect->clear();

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;

	for (unsigned int tag = 0, tag_e = (1U << (ndims * 2)); tag < tag_e; ++tag)
	{
		nTuple<int, 3> coords_shift;

		bool tag_is_valid = true;

		for (int n = 0; n < ndims; ++n)
		{
			if (((tag >> (n * 2)) & 3UL) == 3UL)
			{
				tag_is_valid = false;
				break;
			}

			coords_shift[n] = ((tag >> (n * 2)) & 3U) - 1;

			switch (coords_shift[n])
			{
			case 0:
				send_count[n] = l_count[n];
				send_offset[n] = l_offset[n];
				recv_count[n] = l_count[n];
				recv_offset[n] = l_offset[n];
				break;
			case -1: //left

				send_count[n] = ghost_width[n];
				send_offset[n] = l_offset[n];

				recv_count[n] = ghost_width[n];
				recv_offset[n] = l_offset[n] - ghost_width[n];

				break;
			case 1: //right
				send_count[n] = ghost_width[n];
				send_offset[n] = l_offset[n] + l_count[n] - ghost_width[n];

				recv_count[n] = ghost_width[n];
				recv_offset[n] = l_offset[n] + l_count[n];
				break;
			default:
				tag_is_valid = false;
				break;
			}

			if (send_count[n] == 0 || recv_count[n] == 0)
			{
				tag_is_valid = false;
				break;
			}

		}

		if (tag_is_valid
				&& (coords_shift[0] != 0 || coords_shift[1] != 0
				|| coords_shift[2] != 0))
		{

			dist_connect->emplace_back(dist_sync_connection {coords_shift,
			                                                 send_offset, send_count, recv_offset, recv_count});
		}
	}

}

void DistributedObject::make_send_recv_list(int object_id, DataType const &datatype, int ndims,
		size_t const *l_dims,
		std::vector<dist_sync_connection> const &ghost_shape,
		std::vector<mpi_send_recv_s> *res)
{

	auto &mpi_comm = SingletonHolder<simpla::MPIComm>::instance();

	for (auto const &item : ghost_shape)
	{
		int dest, send_tag, recv_tag;

		std::tie(dest, send_tag, recv_tag) = GLOBAL_COMM.make_send_recv_tag(object_id,
				&item.coord_shift[0]);

		res->emplace_back(

				mpi_send_recv_s {

						dest,

						send_tag,

						recv_tag,

						MPIDataType::create(datatype, ndims, l_dims,
								&item.send_offset[0], nullptr, &item.send_count[0],
								nullptr),

						MPIDataType::create(datatype, ndims, l_dims,
								&item.recv_offset[0], nullptr, &item.recv_count[0],
								nullptr)}

		);

	}
}
//	if (pghost_width == nullptr /*|| mpi_comm.num_of_process() <= 1*/)
//	{
//		return;
//	}
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> ghost_width;
//
//	ghost_width = pghost_width;
//
//	size_t ndims = 3;
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> g_dims, g_offset, g_count;
//
//	std::tie(ndims, g_dims, g_offset, std::ignore, g_count, std::ignore) =
//			dataspace.global_shape();
//
//	nTuple<size_t, 3> l_dims, l_offset, l_count;
//
//	std::tie(std::ignore, l_dims, l_offset, std::ignore, l_count, std::ignore) =
//			dataspace.shape();
//
//	auto mpi_topology = mpi_comm.Topology();
//
//	for (int n = 0; n < ndims; ++n)
//	{
//		if (mpi_topology[n] > (l_dims[n]))
//		{
//			RUNTIME_ERROR(
//					"DataSpace decompose fail! Dimension  is smaller than process grid. "
//							"[dimensions= " + value_to_string(l_dims)
//							+ ", process dimensions="
//							+ value_to_string(mpi_topology) + ", ghost_width"
//							+ value_to_string(ghost_width));
//		}
//	}
//
//	int count = 0;
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;
//
//	for (unsigned int tag = 0, tag_e = (1UL << (ndims * 2)); tag < tag_e; ++tag)
//	{
//		nTuple<int, MAX_NDIMS_OF_ARRAY> coords_shift;
//
//		bool tag_is_valid = false;
//
//		for (int n = 0; n < ndims; ++n)
//		{
//			coords_shift[n] = ((tag >> (n * 2)) & 3UL) - 1;
//
//			switch (coords_shift[n])
//			{
//			case 0:
//				send_count[n] = l_count[n];
//				send_offset[n] = l_offset[n];
//				recv_count[n] = l_count[n];
//				recv_offset[n] = l_offset[n];
//				break;
//			case -1:
//
//				send_count[n] = ghost_width[n];
//				send_offset[n] = l_offset[n];
//				recv_count[n] = ghost_width[n];
//				recv_offset[n] = l_offset[n] - ghost_width[n];
//				tag_is_valid = true;
//				break;
//
//			case 1:
//
//				send_count[n] = ghost_width[n];
//				send_offset[n] = l_offset[n] + l_count[n] - ghost_width[n];
//				recv_count[n] = ghost_width[n];
//				recv_offset[n] = l_offset[n] + l_count[n];
//				tag_is_valid = true;
//				break;
//			}
//
//			if (send_count[n] == 0 || recv_count[n] == 0)
//			{
//				tag_is_valid = false;
//				break;
//			}
//		}
//
//		if (tag_is_valid)
//		{
//
//			res->emplace_back(
//
//					mpi_send_recv_s {
//
//					mpi_comm.get_neighbour(coords_shift),
//
//					static_cast<int>(tag),
//
//					static_cast<int>(tag),
//
//					MPIDataType::create(datatype, ndims, &l_dims[0],
//							&send_offset[0], nullptr, &send_count[0], nullptr),
//
//					MPIDataType::create(datatype, ndims, &l_dims[0],
//							&recv_offset[0], nullptr, &recv_count[0], nullptr) }
//
//					);
//		}
//
//	}


//void sync_update_dataset(DataSet * dset, size_t const * ghost_width,
//		std::vector<MPI_Request> *requests)
//{
//	std::vector<mpi_send_recv_s> s_r_list;
//
//	make_send_recv_list(dset->dataspace, dset->datatype, ghost_width,
//			&s_r_list);
//
//	sync_update_continue(s_r_list, dset->data.get(), requests);
//}

}  // namespace simpla
