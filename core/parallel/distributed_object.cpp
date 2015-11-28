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

    int m_object_id_;

    std::vector<MPI_Request> m_mpi_requests_;
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
    ASSERT(pimpl_ != nullptr);

    if (!GLOBAL_COMM.is_valid()) { return; }

    for (auto const &item :  send_buffer)
    {
        int dest_id, send_tag, recv_tag;

        std::tie(dest_id, send_tag, std::ignore) = GLOBAL_COMM.make_send_recv_tag(pimpl_->m_object_id_,
                                                                                  &std::get<0>(item)[0]);


        MPI_Request req;

        MPI_ERROR(MPI_Isend(std::get<1>(item).data.get(), 1,
                            MPIDataType::create(std::get<1>(item).datatype, std::get<1>(item).memory_space).type(),
                            dest_id,
                            send_tag, GLOBAL_COMM.comm(), &req));

        pimpl_->m_mpi_requests_.push_back(std::move(req));
    }


    for (auto &item : recv_buffer)
    {

        int dest_id, send_tag, recv_tag;

        std::tie(dest_id, std::ignore, recv_tag)
                = GLOBAL_COMM.make_send_recv_tag(pimpl_->m_object_id_, &std::get<0>(item)[0]);

        if (std::get<1>(item).dataspace.size() <= 0 || std::get<1>(item).data == nullptr)
        {
            MPI_Status status;

            MPI_ERROR(MPI_Probe(dest_id, recv_tag, GLOBAL_COMM.comm(), &status));

            // When probe returns, the status object has the size and other
            // attributes of the incoming message. Get the size of the message
            int recv_num = 0;


            MPI_ERROR(MPI_Get_count(&status,
                                    MPIDataType::create(std::get<1>(item).datatype).type(), &recv_num));

            if (recv_num == MPI_UNDEFINED)
            {
                THROW_EXCEPTION_RUNTIME_ERROR("Update Ghosts Particle fail");
            }

            std::get<1>(item).data = sp_alloc_memory(recv_num * std::get<1>(item).datatype.size());

            size_t s_recv_num = recv_num;
            std::get<1>(item).memory_space = DataSpace(1, &s_recv_num);
        }
        MPI_Request req;

        MPI_ERROR(MPI_Irecv(std::get<1>(item).data.get(), 1,
                            MPIDataType::create(std::get<1>(item).datatype, std::get<1>(item).memory_space).type(),
                            dest_id,
                            recv_tag, GLOBAL_COMM.comm(), &req));

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


void DistributedObject::add(DataSet ds)
{

    typedef typename DataSpace::index_tuple index_tuple;

    int ndims;
    index_tuple dimensions;
    index_tuple start;
//    index_tuple stride;
    index_tuple count;
//    index_tuple block;



    std::tie(ndims, dimensions, start, std::ignore, count, std::ignore)
            = ds.memory_space.shape();


    ASSERT(start + count <= dimensions);

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
            try
            {

                DataSet send_buffer(ds);

                DataSet recv_buffer(ds);

                send_buffer.memory_space.select_hyperslab(&send_offset[0], nullptr, &send_count[0], nullptr);

                recv_buffer.memory_space.select_hyperslab(&recv_offset[0], nullptr, &recv_count[0], nullptr);

                this->add_link_send(coord_offset, send_buffer);

                this->add_link_recv(coord_offset, send_buffer);

            }
            catch (std::exception const &error)
            {
                THROW_EXCEPTION_RUNTIME_ERROR("add coommnication link error", error.what());

            }
        }

    }

}


}}//namespace simpla{ namespace parallel