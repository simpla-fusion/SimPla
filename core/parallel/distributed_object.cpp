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


	void add_link(bool is_send, int const coord_offset[], int size,
			MPIDataType const &d_type, std::shared_ptr<void> *p);

};

//! Default constructor
DistributedObject::DistributedObject(MPIComm &comm)
		: pimpl_(new pimpl_s(comm))
{

}

DistributedObject::DistributedObject(DistributedObject const &other) : pimpl_(new pimpl_s(*other.pimpl_))
{

}

DistributedObject::~DistributedObject()
{

}

DistributedObject::pimpl_s::pimpl_s(MPIComm &comm) : m_comm_(comm), m_object_id_(m_comm_.generate_object_id())
{

}

MPIComm &DistributedObject::comm() const
{
	return pimpl_->m_comm_;
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

void DistributedObject::pimpl_s::add_link(bool is_send, int const coord_offset[], int size,
		MPIDataType const &mpi_d_type, std::shared_ptr<void> *p)
{
	int dest_id, send_tag, recv_tag;

	std::tie(dest_id, send_tag, recv_tag) = m_comm_.make_send_recv_tag(m_object_id_, &coord_offset[0]);

	if (is_send)
	{
		m_send_links_.push_back(mpi_link_node({dest_id, send_tag, size, mpi_d_type, p}));
	}
	else
	{
		m_recv_links_.push_back(mpi_link_node({dest_id, recv_tag, size, mpi_d_type, p}));
	}
}

void DistributedObject::add_link(bool is_send, int const coord_offset[], int size,
		DataType const &d_type, std::shared_ptr<void> *p)
{
	pimpl_->add_link(is_send, coord_offset, size, MPIDataType::create(d_type), p);


}

void DistributedObject::add_link(bool is_send, int const coord_offset[], DataSpace d_space,
		DataType const &d_type, std::shared_ptr<void> *p)
{

	pimpl_->add_link(is_send, coord_offset, 1, MPIDataType::create(d_type, d_space), p);


}


}//namespace simpla