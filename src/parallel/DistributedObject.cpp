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
    typedef typename data_model::DataSpace::index_tuple index_tuple;

    pimpl_s();

    pimpl_s(pimpl_s const &) = delete;

    ~pimpl_s();


    void sync();

    void wait();

    void clear();

    bool is_ready() const;


    struct link_s
    {
        bool only_once;
        int tag;
        int dest;
        nTuple<ptrdiff_t, 3> shift;
        data_model::DataSet data_set;
    };

    std::multimap<size_t, link_s> m_send_links_;
    std::multimap<size_t, link_s> m_recv_links_;

    std::vector<MPIDataType> m_mpi_dtype_;
    std::vector<MPI_Request> m_mpi_requests_;

    void add(int id, data_model::DataSet ds, bool only_once);


    void add_send_link(size_t id, const nTuple<ptrdiff_t, 3> &shift, data_model::DataSet ds,
                       bool only_once = false);

    void add_recv_link(size_t id, const nTuple<ptrdiff_t, 3> &shift, data_model::DataSet ds, bool only_once = false);

    void remove(int);


};

DistributedObject::pimpl_s::pimpl_s() { }

DistributedObject::pimpl_s::~pimpl_s() { }

void DistributedObject::pimpl_s::clear()
{
    m_send_links_.clear();
    m_recv_links_.clear();

    m_mpi_dtype_.clear();
    m_mpi_requests_.clear();
}

void DistributedObject::pimpl_s::add_send_link(size_t id,
                                               const nTuple<ptrdiff_t, 3> &shift,
                                               data_model::DataSet ds,
                                               bool only_once)
{
    int dest_id;
    int send_tag;
    std::tie(dest_id, send_tag, std::ignore) = GLOBAL_COMM.make_send_recv_tag(id, shift);

    m_send_links_.emplace(
            std::make_pair(id, link_s{only_once, send_tag, dest_id,
                                      {shift[0], shift[1], shift[2]},
                                      std::move(ds)}));


};

void DistributedObject::pimpl_s::add_recv_link(size_t id,
                                               const nTuple<ptrdiff_t, 3> &shift,
                                               data_model::DataSet ds,
                                               bool only_once)
{
    int dest_id;
    int recv_tag;
    std::tie(dest_id, recv_tag, std::ignore) = GLOBAL_COMM.make_send_recv_tag(id, shift);

    m_recv_links_.emplace(
            std::make_pair(id, link_s{only_once, recv_tag, dest_id,
                                      {shift[0], shift[1], shift[2]},
                                      std::move(ds)}));

}

void DistributedObject::pimpl_s::remove(int id)
{
    m_send_links_.erase(id);
    m_recv_links_.erase(id);
};


void DistributedObject::pimpl_s::add(int id, data_model::DataSet ds, bool only_once = false)
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


    nTuple<ptrdiff_t, 3> send_offset;
    nTuple<size_t, 3> send_count;
    nTuple<ptrdiff_t, 3> recv_offset;
    nTuple<size_t, 3> recv_count;

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

                data_model::DataSet send_ds(ds);

                data_model::DataSet recv_ds(ds);

                send_ds.memory_space.select_hyperslab(&send_offset[0], nullptr, &send_count[0], nullptr);

                recv_ds.memory_space.select_hyperslab(&recv_offset[0], nullptr, &recv_count[0], nullptr);


                add_send_link(id, send_offset, std::move(send_ds), only_once);

                add_send_link(id, recv_offset, std::move(recv_ds), only_once);


            }
            catch (std::exception const &error)
            {
                RUNTIME_ERROR << "add communication link error" << error.what() << std::endl;

            }
        }

    }

}


void DistributedObject::pimpl_s::sync()
{

    if (!GLOBAL_COMM.is_valid()) { return; }

    m_mpi_requests_.clear();
    m_mpi_requests_.resize(m_send_links_.size() + m_recv_links_.size());
    m_mpi_dtype_.clear();
    m_mpi_dtype_.resize(m_send_links_.size() + m_recv_links_.size());


    size_t count = 0;
    for (auto const &item :  m_send_links_)
    {

        ASSERT(item.second.data_set.data != nullptr);
        MPIDataType::create(item.second.data_set.data_type,
                            item.second.data_set.memory_space).
                swap(m_mpi_dtype_[count]);

        MPI_ERROR(MPI_Isend(item.second.data_set.data.get(), 1,
                            m_mpi_dtype_[count].type(),
                            item.second.dest, item.second.tag,
                            GLOBAL_COMM.comm(),
                            &(m_mpi_requests_[item.first])));
        ++count;
    }


    for (auto &item : m_recv_links_)
    {

        int dest_id = item.second.dest;

        int recv_tag = item.second.tag;

        auto &ds = item.second.data_set;


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

//            ds.data_space = ds.memory_space;

            MPIDataType::create(ds.data_type).swap(m_mpi_dtype_[count]);

            ASSERT(ds.data.get() != nullptr);
            MPI_ERROR(MPI_Irecv(ds.data.get(), recv_num,
                                m_mpi_dtype_[count].type(),
                                dest_id, recv_tag, GLOBAL_COMM.comm(),
                                &(m_mpi_requests_[count])));

        }
        else
        {
            ASSERT(ds.data.get() != nullptr);
            MPIDataType::create(ds.data_type, ds.memory_space).swap(m_mpi_dtype_[count]);
            MPI_ERROR(MPI_Irecv(ds.data.get(), 1,
                                m_mpi_dtype_[count].type(),
                                dest_id, recv_tag, GLOBAL_COMM.comm(),
                                &(m_mpi_requests_[count])));
        }

        ++count;
    }

 
}

void DistributedObject::pimpl_s::wait()
{
    if (GLOBAL_COMM.is_valid() && !m_mpi_requests_.empty())
    {

        int count = static_cast<int>(m_mpi_requests_.size());

        MPI_Request *array_of_requests = &(m_mpi_requests_[0]);

        MPI_Waitall(count, array_of_requests, MPI_STATUSES_IGNORE);

        m_mpi_dtype_.clear();
        m_mpi_requests_.clear();
    }


}

bool DistributedObject::pimpl_s::is_ready() const
{
    //! FIXME this is not multi-threads safe

    if (m_mpi_requests_.size() > 0)
    {
        int flag = 0;

        int count = static_cast<int>(m_mpi_requests_.size());

        MPI_Request *array_of_requests = &(const_cast<pimpl_s *>(this)->m_mpi_requests_[0]);

        MPI_ERROR(MPI_Testall(count, array_of_requests, &flag, MPI_STATUSES_IGNORE));

        return flag != 0;
    }

    return true;

}


//! Default constructor
DistributedObject::DistributedObject() : pimpl_(new pimpl_s()) { }


DistributedObject::~DistributedObject() { }


void DistributedObject::clear() { pimpl_->clear(); }

void DistributedObject::sync() { pimpl_->sync(); }

void DistributedObject::wait() { pimpl_->wait(); }

bool DistributedObject::is_ready() const { pimpl_->is_ready(); }

void DistributedObject::add(size_t id, data_model::DataSet &ds, bool only_once)
{
    pimpl_->add(id, ds, only_once);
}

void DistributedObject::add_send_link(size_t id, const nTuple<ptrdiff_t, 3> &offset, data_model::DataSet ds)
{
    return pimpl_->add_send_link(id, offset, ds);

};

void DistributedObject::add_recv_link(size_t id, const nTuple<ptrdiff_t, 3> &offset, data_model::DataSet ds)
{
    return pimpl_->add_recv_link(id, offset, ds);
}

void DistributedObject::remove(size_t id)
{
    pimpl_->remove(id);
}

}}//namespace simpla{ namespace parallel