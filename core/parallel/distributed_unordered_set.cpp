/**
 * @file distributed_unordered_set.cpp
 * @author salmon
 * @date 2015-10-15.
 */

#include "distributed_unordered_set.h"


#include "../gtl/utilities/log.h"
#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_update.h"

namespace simpla
{
struct DistributedUnorderedSetBase::pimpl_s
{
	std::vector<mpi_send_recv_s> m_send_recv_list_;
	std::vector<mpi_send_recv_buffer_s> m_send_recv_buffer_;
	std::vector<MPI_Request> m_mpi_requests_;


};

//! Default constructor
DistributedUnorderedSetBase::DistributedUnorderedSetBase() : pimpl_(new pimpl_s)
{
}


//! destroy.
DistributedUnorderedSetBase::~DistributedUnorderedSetBase()
{
}

bool DistributedUnorderedSetBase::is_ready() const
{
	//! FIXME this is not multi-threads safe

	if (!is_valid())
	{
		return false;
	}

	if (pimpl_->m_mpi_requests_.size() > 0)
	{
		int flag = 0;
		MPI_ERROR(MPI_Testall(static_cast<int>( pimpl_->m_mpi_requests_.size()), //
				const_cast<MPI_Request *>(&pimpl_->m_mpi_requests_[0]),//
				&flag, MPI_STATUSES_IGNORE));

		return flag != 0;
	}

	return true;

}


void DistributedUnorderedSetBase::deploy(std::vector<dist_sync_connection> const &ghost_shape)
{
	int ndims = 3;

	auto ds = dataset();

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> l_dims;

	auto d_shape = ds.dataspace.local_shape();

	ndims = d_shape.ndims;

	l_dims = d_shape.dimensions;

	make_send_recv_list(global_id(), ds.datatype, ndims, &l_dims[0],
			ghost_shape, &m_send_recv_list_);
}

void DistributedUnorderedSetBase::sync()
{
	auto ghost_list = m_domain_.mesh().template ghost_shape<iform>();

	for (auto const &item : ghost_list)
	{
		mpi_send_recv_buffer_s send_recv_s;

		send_recv_s.datatype = traits::datatype<value_type>::create();

		std::tie(send_recv_s.dest, send_recv_s.send_tag,
				send_recv_s.recv_tag) = GLOBAL_COMM.make_send_recv_tag(global_id(),
				&item.coord_shift[0]);

		//   collect send data

		domain_type send_range(m_domain_);

		send_range.select(item.send_offset,
				item.send_offset + item.send_count);

		send_recv_s.send_size = container_type::size_all(send_range);

		send_recv_s.send_data = sp_alloc_memory(
				send_recv_s.send_size * send_recv_s.datatype.size());

		value_type *data =
				reinterpret_cast<value_type *>(send_recv_s.send_data.get());

		// FIXME need parallel optimize
		for (auto const &key : send_range)
		{
			for (auto const &p : container_type::operator[](key))
			{
				*data = p;
				++data;
			}
		}

		send_recv_s.recv_size = 0;
		send_recv_s.recv_data = nullptr;
		m_send_recv_buffer_.push_back(std::move(send_recv_s));

		//  clear ghosts cell
		domain_type recv_range(m_domain_);

		recv_range.select(item.recv_offset, item.recv_offset + item.recv_count);

		container_type::erase(recv_range);

	}

	sync_update_varlength(&m_send_recv_buffer_, &(m_mpi_requests_));
}


void DistributedUnorderedSetBase::wait() const
{
	if (GLOBAL_COMM.is_valid() && !pimpl_->m_mpi_requests_.empty())
	{
		wait_all_request(const_cast<std::vector<MPI_Request> *>(&pimpl_->m_mpi_requests_));
	}
}

void DistributedUnorderedSetBase::make_send_recv_list(int object_id, DataType const &datatype, int ndims,
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
//					mpi_send_recv_block_s {
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
//	std::vector<mpi_send_recv_block_s> s_r_list;
//
//	make_send_recv_list(dset->dataspace, dset->datatype, ghost_width,
//			&s_r_list);
//
//	sync_update_block(s_r_list, dset->data.get(), requests);
//}

}  // namespace simpla
