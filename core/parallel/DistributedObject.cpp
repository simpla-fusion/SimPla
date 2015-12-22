/**
 * @file distributed_object.cpp
 * @author salmon
 * @date 2015-10-17.
 */

#include "DistributedObject.h"

#include "MPIComm.h"
#include "MPIAuxFunctions.h"
#include "MPIUpdate.h"

namespace simpla { namespace parallel
{

struct DistributedObject::pimpl_s
{
    pimpl_s();

    pimpl_s(pimpl_s const &) = delete;

    ~pimpl_s() { }

    int m_object_id_;


    std::vector<MPIDataType> m_mpi_dtype_;
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

    pimpl_->m_mpi_requests_.clear();
    pimpl_->m_mpi_requests_.resize(send_buffer.size() + recv_buffer.size());
    pimpl_->m_mpi_dtype_.clear();
    pimpl_->m_mpi_dtype_.resize(send_buffer.size() + recv_buffer.size());

    int count = 0;

    for (auto const &item :  send_buffer)
    {
        int dest_id, send_tag, recv_tag;

        std::tie(dest_id, send_tag, std::ignore) = GLOBAL_COMM.make_send_recv_tag(pimpl_->m_object_id_,
                                                                                  &std::get<0>(item)[0]);
        auto &ds = std::get<1>(item);
        ASSERT(ds.data != nullptr);
        MPIDataType::create(ds.data_type, ds.memory_space).swap(pimpl_->m_mpi_dtype_[count]);
        MPI_ERROR(MPI_Isend(ds.data.get(), 1,
                            pimpl_->m_mpi_dtype_[count].type(),
                            dest_id, send_tag, GLOBAL_COMM.comm(), &(pimpl_->m_mpi_requests_[count])));

        ++count;
    }


    for (auto &item : recv_buffer)
    {

        int dest_id, send_tag, recv_tag;

        std::tie(dest_id, std::ignore, recv_tag)
                = GLOBAL_COMM.make_send_recv_tag(pimpl_->m_object_id_, &std::get<0>(item)[0]);

        auto &ds = std::get<1>(item);


        if (ds.memory_space.size() <= 0 || ds.data == nullptr)
        {
            MPI_Status status;

            MPI_ERROR(MPI_Probe(dest_id, recv_tag, GLOBAL_COMM.comm(), &status));

            // When probe returns, the status object has the size and other
            // attributes of the incoming message. Get the size of the message
            int recv_num = 0;


            auto m_dtype = MPIDataType::create(ds.data_type);

            MPI_ERROR(MPI_Get_count(&status, m_dtype.type(), &recv_num));

            if (recv_num == MPI_UNDEFINED)
            {
                THROW_EXCEPTION_RUNTIME_ERROR("Update Ghosts Particle fail");
            }

            ds.data = sp_alloc_memory(recv_num * ds.data_type.size_in_byte());

            size_t s_recv_num = static_cast<size_t>(recv_num);

            ds.memory_space = data_model::DataSpace(1, &s_recv_num);

            ds.data_space = ds.memory_space;

            MPIDataType::create(ds.data_type).swap(pimpl_->m_mpi_dtype_[count]);

            ASSERT(ds.data.get() != nullptr);
            MPI_ERROR(MPI_Irecv(ds.data.get(), recv_num,
                                pimpl_->m_mpi_dtype_[count].type(),
                                dest_id, recv_tag, GLOBAL_COMM.comm(),
                                &(pimpl_->m_mpi_requests_[count])));

        }
        else
        {
            ASSERT(ds.data.get() != nullptr);
            MPIDataType::create(ds.data_type, ds.memory_space).swap(pimpl_->m_mpi_dtype_[count]);
            MPI_ERROR(MPI_Irecv(ds.data.get(), 1,
                                pimpl_->m_mpi_dtype_[count].type(),
                                dest_id, recv_tag, GLOBAL_COMM.comm(),
                                &(pimpl_->m_mpi_requests_[count])));
        }


        ++count;
    }


}

void DistributedObject::wait()
{
    if (GLOBAL_COMM.is_valid() && !pimpl_->m_mpi_requests_.empty())
    {

        int count = static_cast<int>(pimpl_->m_mpi_requests_.size());

        MPI_Request *array_of_requests = &(pimpl_->m_mpi_requests_[0]);


        MPI_Waitall(count, array_of_requests, MPI_STATUSES_IGNORE);


        pimpl_->m_mpi_dtype_.clear();
        pimpl_->m_mpi_requests_.clear();
    }


}

bool DistributedObject::is_ready() const
{
    //! FIXME this is not multi-threads safe

    if (pimpl_->m_mpi_requests_.size() > 0)
    {
        int flag = 0;

        int count = static_cast<int>(pimpl_->m_mpi_requests_.size());

        MPI_Request *array_of_requests = &(pimpl_->m_mpi_requests_[0]);

        MPI_ERROR(MPI_Testall(count, array_of_requests, &flag, MPI_STATUSES_IGNORE));

        return flag != 0;
    }

    return true;

}


void DistributedObject::add(data_model::DataSet ds)
{

    typedef typename data_model::DataSpace::index_tuple index_tuple;

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

                data_model::DataSet send_buffer(ds);

                data_model::DataSet recv_buffer(ds);

                send_buffer.memory_space.select_hyperslab(&send_offset[0], nullptr, &send_count[0], nullptr);

                recv_buffer.memory_space.select_hyperslab(&recv_offset[0], nullptr, &recv_count[0], nullptr);

                this->add_link_send(coord_offset, send_buffer);

                this->add_link_recv(coord_offset, recv_buffer);

            }
            catch (std::exception const &error)
            {
                RUNTIME_ERROR << "add coommnication link error" << error.what() << std::endl;

            }
        }

    }

}


}}//namespace simpla{ namespace parallel