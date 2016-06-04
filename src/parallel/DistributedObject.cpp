/**
 * @file distributed_object.cpp
 * @author salmon
 * @date 2015-10-17.
 */

#include "DistributedObject.h"

#include "MPIComm.h"
#include "MPIAuxFunctions.h"
#include "MPIUpdate.h"

namespace simpla
{
namespace parallel
{

struct DistributedObject::pimpl_s
{
    pimpl_s();

    pimpl_s(pimpl_s const &) = delete;

    ~pimpl_s();


    void sync();

    void wait();

    void clear();

    bool is_ready() const;


    struct link_s
    {
        int id;
        int dest_id;
        nTuple<int, 3> shift;
        data_model::DataSet data_set;
    };

    std::map<int, link_s> m_send_links_;
    std::map<int, link_s> m_recv_links_;

    std::map<int, MPIDataType> m_mpi_dtype_;
    std::map<int, MPI_Request> m_mpi_requests_;

    void add(int id, data_model::DataSet ds, std::vector<int> *_send_tag, std::vector<int> *_recv_tag);

    int add_send_link(int id, const int *shift, data_model::DataSet ds);

    int add_recv_link(int id, const int *i, data_model::DataSet ds);

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

int DistributedObject::pimpl_s::add_send_link(int id, const int *shift, data_model::DataSet ds)
{
    int dest_id;
    int send_tag;
    std::tie(dest_id, send_tag, std::ignore) = GLOBAL_COMM.make_send_recv_tag(id, shift);

    m_send_links_.emplace(
            std::make_pair(send_tag, link_s{id, dest_id, nTuple<int, 3>{shift[0], shift[1], shift[2]}, std::move(ds)}));

    return send_tag;

};

int DistributedObject::pimpl_s::add_recv_link(int id, const int *shift, data_model::DataSet ds)
{
    int dest_id;
    int recv_tag;
    std::tie(dest_id, recv_tag, std::ignore) = GLOBAL_COMM.make_send_recv_tag(id, shift);

    m_recv_links_.emplace(
            std::make_pair(recv_tag, link_s{id, dest_id, nTuple<int, 3>{shift[0], shift[1], shift[2]}, std::move(ds)}));

    return recv_tag;


}

void DistributedObject::pimpl_s::sync()
{

    if (!GLOBAL_COMM.is_valid()) { return; }

    m_mpi_requests_.clear();
    m_mpi_dtype_.clear();


    for (auto const &item :  m_send_links_)
    {

        ASSERT(item.second.data_set.data != nullptr);
        MPIDataType::create(item.second.data_set.data_type,
                            item.second.data_set.memory_space).
                swap(m_mpi_dtype_[item.first]);

        MPI_ERROR(MPI_Isend(item.second.data_set.data.get(), 1,
                            m_mpi_dtype_[item.first].type(),
                            item.second.dest_id, item.first,
                            GLOBAL_COMM.comm(),
                            &(m_mpi_requests_[item.first])));

    }


    for (auto &item : m_recv_links_)
    {

        int dest_id = item.second.dest_id;

        int recv_tag = item.first;

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

            MPIDataType::create(ds.data_type).swap(m_mpi_dtype_[recv_tag]);

            ASSERT(ds.data.get() != nullptr);
            MPI_ERROR(MPI_Irecv(ds.data.get(), recv_num,
                                m_mpi_dtype_[recv_tag].type(),
                                dest_id, recv_tag, GLOBAL_COMM.comm(),
                                &(m_mpi_requests_[recv_tag])));

        }
        else
        {
            ASSERT(ds.data.get() != nullptr);
            MPIDataType::create(ds.data_type, ds.memory_space).swap(m_mpi_dtype_[recv_tag]);
            MPI_ERROR(MPI_Irecv(ds.data.get(), 1,
                                m_mpi_dtype_[recv_tag].type(),
                                dest_id, recv_tag, GLOBAL_COMM.comm(),
                                &(m_mpi_requests_[recv_tag])));
        }


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


void DistributedObject::pimpl_s::add(int id, data_model::DataSet ds, std::vector<int> *_send_tag,
                                     std::vector<int> *_recv_tag)
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

                data_model::DataSet send_ds(ds);

                data_model::DataSet recv_ds(ds);

                send_ds.memory_space.select_hyperslab(&send_offset[0], nullptr, &send_count[0], nullptr);

                recv_ds.memory_space.select_hyperslab(&recv_offset[0], nullptr, &recv_count[0], nullptr);

                nTuple<int, 3> s_offset, r_offset;

                s_offset = send_offset;

                r_offset = recv_offset;


                auto s_t = add_send_link(id, &s_offset[0], std::move(send_ds));

                auto r_t = add_send_link(id, &r_offset[0], std::move(recv_ds));

                if (_send_tag != nullptr) { _send_tag->push_back(s_t); }
                if (_recv_tag != nullptr) { _recv_tag->push_back(r_t); }

            }
            catch (std::exception const &error)
            {
                RUNTIME_ERROR << "add communication link error" << error.what() << std::endl;

            }
        }

    }

}


//! Default constructor
DistributedObject::DistributedObject() : pimpl_(new pimpl_s()) { }


DistributedObject::~DistributedObject() { }


void DistributedObject::clear() { pimpl_->clear(); }

void DistributedObject::sync() { pimpl_->sync(); }

void DistributedObject::wait() { pimpl_->wait(); }

bool DistributedObject::is_ready() const { pimpl_->is_ready(); }

void DistributedObject::add(int id, data_model::DataSet &ds, std::vector<int> *_send_tag,
                            std::vector<int> *_recv_tag)
{
    pimpl_->add(id, ds, _send_tag, _recv_tag);
}

void DistributedObject::remove(int tag, bool is_recv)
{
    if (is_recv) { pimpl_->m_recv_links_.erase(tag); }
    else { pimpl_->m_send_links_.erase(tag); }
}

int DistributedObject::add_send_link(int id, const int offset[3], data_model::DataSet ds)
{
    return pimpl_->add_send_link(id, offset, ds);

};

int DistributedObject::add_recv_link(int id, const int offset[3], data_model::DataSet ds)
{
    return pimpl_->add_recv_link(id, offset, ds);
}


}
}//namespace simpla{ namespace parallel