/**
 * @file distributed_object.cpp
 * @author salmon
 * @date 2015-10-17.
 */

#include "distributed_object.h"

#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_update.h"

namespace simpla { namespace parallel
{

struct DistributedObject::pimpl_s
{
    pimpl_s();

    pimpl_s(pimpl_s const &) = delete;

    ~pimpl_s() { }


    struct mpi_link_node
    {
        int dest_id;

        int tag;

        int size;

        MPIDataType type;

        std::shared_ptr<void> *data; // pointer is a risk

    };


    int m_object_id_;

    std::vector<mpi_link_node> m_send_links_;

    std::vector<mpi_link_node> m_recv_links_;

    std::vector<MPI_Request> m_mpi_requests_;


    void add_link(bool is_send, int const coord_offset[], int size,
                  MPIDataType const &d_type, std::shared_ptr<void> *p);

};

//! Default constructor
DistributedObject::DistributedObject()
        : pimpl_(new pimpl_s())
{
}


DistributedObject::~DistributedObject()
{

}

DistributedObject::pimpl_s::pimpl_s() : m_object_id_(GLOBAL_COMM.generate_object_id())
{

}


void DistributedObject::sync()
{
    if (!GLOBAL_COMM.is_valid())
    {
        return;
    }
    for (auto const &item : pimpl_->m_send_links_)
    {
        MPI_Request req;

        MPI_ERROR(MPI_Isend(item.data->get(), item.size, item.type.type(), item.dest_id,
                            item.tag, GLOBAL_COMM.comm(), &req));

        pimpl_->m_mpi_requests_.push_back(std::move(req));
    }


    for (auto &item : pimpl_->m_recv_links_)
    {
        if (item.size <= 0 || item.data == nullptr)
        {
            MPI_Status status;

            MPI_ERROR(MPI_Probe(item.dest_id, item.tag, GLOBAL_COMM.comm(), &status));

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
                            item.tag, GLOBAL_COMM.comm(), &req));

        pimpl_->m_mpi_requests_.push_back(std::move(req));
    }


}

void DistributedObject::wait()
{

    if (GLOBAL_COMM.is_valid() && !pimpl_->m_mpi_requests_.empty())
    {

        MPI_ERROR(MPI_Waitall(static_cast<int>(pimpl_->m_mpi_requests_.size()),
                              const_cast<MPI_Request *>(&(pimpl_->m_mpi_requests_[0])),
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

    std::tie(dest_id, send_tag, recv_tag) = GLOBAL_COMM.make_send_recv_tag(m_object_id_, &coord_offset[0]);

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

void DistributedObject::add_link(bool is_send, int const coord_offset[], DataSpace const &d_space,
                                 DataType const &d_type, std::shared_ptr<void> *p)
{
    pimpl_->add_link(is_send, coord_offset, 1, MPIDataType::create(d_type, d_space), p);
}


void DistributedObject::add(DataSet ds)
{

    typedef typename DataSpace::index_tuple index_tuple;

    int ndims;
    index_tuple dimensions;
    index_tuple start;
//    index_tuple stride;
    index_tuple count;
//    index_tuple block;

    std::tie(ndims, dimensions, start, std::ignore, count, std::ignore) = ds.memory_space.shape();

    index_tuple ghost_width = start;


    index_tuple send_offset, send_count;
    index_tuple recv_offset, recv_count;

    for (unsigned int tag = 0, tag_e = (1U << (ndims * 2)); tag < tag_e; ++tag)
    {
        nTuple<int, 3> coord_offset;

        bool tag_is_valid = true;

        for (int n = 0; n < ndims; ++n)
        {
            if (((tag >> (n * 2)) & 3UL) == 3UL)
            {
                tag_is_valid = false;
                break;
            }

            coord_offset[n] = ((tag >> (n * 2)) & 3U) - 1;

            switch (coord_offset[n])
            {
                case 0:
                    send_offset[n] = start[n];
                    send_count[n] = count[n];
                    recv_offset[n] = start[n];
                    recv_count[n] = count[n];

                    break;
                case -1: //left

                    send_offset[n] = start[n];
                    send_count[n] = ghost_width[n];
                    recv_offset[n] = start[n] - ghost_width[n];
                    recv_count[n] = ghost_width[n];


                    break;
                case 1: //right
                    send_offset[n] = start[n] + count[n] - ghost_width[n];
                    send_count[n] = ghost_width[n];
                    recv_offset[n] = start[n] + count[n];
                    recv_count[n] = ghost_width[n];

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

        if (tag_is_valid && (coord_offset[0] != 0 || coord_offset[1] != 0 || coord_offset[2] != 0))
        {
            this->add_link_send(
                    &coord_offset[0],
                    DataSpace(ds.memory_space).select_hyperslab(&send_offset[0], nullptr, &send_count[0], nullptr),
                    ds.datatype, &ds.data);

            this->add_link_recv(
                    &coord_offset[0],
                    DataSpace(ds.memory_space).select_hyperslab(&recv_offset[0], nullptr, &recv_count[0], nullptr),
                    ds.datatype, &ds.data);

        }

    }

}


}}//namespace simpla{ namespace parallel