/**
 * @file distributed_object.cpp
 * @author salmon
 * @date 2015-10-17.
 */

#include "distributed_object.h"

#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_update.h"

namespace simpla
{
struct DistributedObject::pimpl_s
{
	pimpl_s(MPIComm &);

	pimpl_s(pimpl_s const &) = default;

	~pimpl_s() { }


	struct mpi_link_node
	{
		int dest_id;

		int tag;

		int size;

		MPIDataType type;

		std::shared_ptr<void> *data; // pointer is a risk

	};

	MPIComm &m_comm_;

	int m_object_id_;

	std::vector<mpi_link_node> m_send_links_;

	std::vector<mpi_link_node> m_recv_links_;

	std::vector<MPI_Request> m_mpi_requests_;

};

//! Default constructor
DistributedObject::DistributedObject()
		: pimpl_(new pimpl_s(SingletonHolder<MPIComm>::instance()))
{

}

//DistributedObject::DistributedObject(DistributedObject const &other) : pimpl_(new pimpl_s)
//{
//
//}

DistributedObject::~DistributedObject()
{

}

DistributedObject::pimpl_s::pimpl_s(MPIComm &comm) : m_comm_(comm), m_object_id_(m_comm_.generate_object_id())
{

}

void DistributedObject::clear_links()
{
	wait();
	pimpl_->m_send_links_.clear();
	pimpl_->m_recv_links_.clear();
}

bool DistributedObject::is_distributed() const
{
	return pimpl_->m_send_links_.size() + pimpl_->m_recv_links_.size() > 0;
}

void DistributedObject::sync()
{

	for (auto const &item : pimpl_->m_send_links_)
	{
		MPI_Request req;

		MPI_ERROR(MPI_Isend(item.data->get(), item.size, item.type.type(), item.dest_id,
				item.tag, pimpl_->m_comm_.comm(), &req));

		pimpl_->m_mpi_requests_.push_back(std::move(req));
	}


	for (auto &item : pimpl_->m_recv_links_)
	{
		if (item.size <= 0 || item.data == nullptr)
		{
			MPI_Status status;

			MPI_ERROR(MPI_Probe(item.dest_id, item.tag, pimpl_->m_comm_.comm(), &status));

			// When probe returns, the status object has the size and other
			// attributes of the incoming message. Get the size of the message
			int recv_num = 0;

			MPI_ERROR(MPI_Get_count(&status, item.type.type(), &recv_num));

			if (recv_num == MPI_UNDEFINED)
			{
				RUNTIME_ERROR("Update Ghosts Particle fail");
			}

			*item.data = sp_alloc_memory(recv_num * item.type.size());

			item.size = recv_num;
		}
		MPI_Request req;

		MPI_ERROR(MPI_Irecv(item.data->get(), item.size, item.type.type(), item.dest_id,
				item.tag, pimpl_->m_comm_.comm(), &req));

		pimpl_->m_mpi_requests_.push_back(std::move(req));
	}


}

void DistributedObject::wait()
{
	deploy();

	if (pimpl_->m_comm_.is_valid() && !pimpl_->m_mpi_requests_.empty())
	{

		MPI_ERROR(MPI_Waitall(pimpl_->m_mpi_requests_.size(), const_cast<MPI_Request *>(&(pimpl_->m_mpi_requests_[0])),
				MPI_STATUSES_IGNORE));

		pimpl_->m_mpi_requests_.clear();

	}
}

bool DistributedObject::is_ready() const
{
	//! FIXME this is not multi-threads safe

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

void DistributedObject::add_link(bool is_send, int const coord_offset[], int size,
		DataType const &d_type, std::shared_ptr<void> *p)
{

	int dest_id, send_tag, recv_tag;

	std::tie(dest_id, send_tag, recv_tag) = pimpl_->m_comm_.make_send_recv_tag(pimpl_->m_object_id_, &coord_offset[0]);

	if (is_send)
	{
		pimpl_->m_send_links_.push_back(
				pimpl_s::mpi_link_node({dest_id, send_tag, size, MPIDataType::create(d_type), p}));
	}
	else
	{
		pimpl_->m_recv_links_.push_back(
				pimpl_s::mpi_link_node({dest_id, recv_tag, size, MPIDataType::create(d_type), p}));
	}
}


void DistributedObject::add_link(DataSet &ds)
{


	auto global_shape = ds.dataspace.global_shape();

	auto local_shape = ds.dataspace.local_shape();

	int ndims = global_shape.ndims;

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> l_dims, l_offset, l_stride, l_count, l_block, ghost_width;

	l_dims = local_shape.dimensions;
	l_offset = local_shape.offset;
	l_stride = local_shape.stride;
	l_count = local_shape.count;
	l_block = local_shape.block;

	ghost_width = l_offset;

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;

	for (unsigned int tag = 0, tag_e = (1U << (ndims * 2)); tag < tag_e; ++tag)
	{
		nTuple<int, 3> coord_shift;

		bool tag_is_valid = true;

		for (int n = 0; n < ndims; ++n)
		{
			if (((tag >> (n * 2)) & 3UL) == 3UL)
			{
				tag_is_valid = false;
				break;
			}

			coord_shift[n] = ((tag >> (n * 2)) & 3U) - 1;

			switch (coord_shift[n])
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

		if (tag_is_valid && (coord_shift[0] != 0 || coord_shift[1] != 0 || coord_shift[2] != 0))
		{

			int dest_id, send_tag, recv_tag;

			std::tie(dest_id, send_tag, recv_tag) = pimpl_->m_comm_.make_send_recv_tag(pimpl_->m_object_id_,
					&coord_shift[0]);


			pimpl_->m_send_links_.push_back(
					pimpl_s::mpi_link_node({dest_id, send_tag, 1,
					                        MPIDataType::create(ds.datatype, ndims,
							                        &l_dims[0], &send_offset[0], nullptr,
							                        &send_count[0], nullptr), &ds.data}));

			pimpl_->m_recv_links_.push_back(
					pimpl_s::mpi_link_node({dest_id, recv_tag, 1,
					                        MPIDataType::create(ds.datatype, ndims,
							                        &l_dims[0], &recv_offset[0], nullptr,
							                        &recv_count[0], nullptr), &ds.data}));


		}
	}

}

}//namespace simpla