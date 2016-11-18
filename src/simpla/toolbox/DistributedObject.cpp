/**
 * @file distributed_object.cpp
 * @author salmon
 * @date 2015-10-17.
 */
#include <simpla/SIMPLA_config.h>

#include "DistributedObject.h"

#include "MemoryPool.h"

#include "MPIComm.h"
#include "MPIAuxFunctions.h"
#include "MPIUpdate.h"

namespace simpla { namespace toolbox
{

struct DistributedObject::pimpl_s
{
    typedef typename data::DataSpace::index_tuple index_tuple;

    pimpl_s();

    pimpl_s(pimpl_s const &) = delete;

    ~pimpl_s();


    void sync();

    void wait();

    void clear();

    bool is_ready() const;


    struct recv_link_s
    {
        int tag;
        int dest;
        nTuple<int, 3> shift;
        data::DataSet *data_set;
    };
    struct send_link_s
    {
        int tag;
        int dest;
        nTuple<int, 3> shift;
        data::DataSet const *data_set;
    };

    std::multimap<size_t, send_link_s> m_send_links_;
    std::multimap<size_t, recv_link_s> m_recv_links_;

    std::vector<MPIDataType> m_mpi_dtype_;
    std::vector<MPI_Request> m_mpi_requests_;


    void add_send_link(size_t id, const nTuple<int, 3> &shift, const toolbox::DataSet *ds);

    void add_recv_link(size_t id, const nTuple<int, 3> &shift, toolbox::DataSet *ds);


};

DistributedObject::pimpl_s::pimpl_s() {}

DistributedObject::pimpl_s::~pimpl_s() {}

void DistributedObject::pimpl_s::clear()
{
    m_send_links_.clear();
    m_recv_links_.clear();

    m_mpi_dtype_.clear();
    m_mpi_requests_.clear();
}

void DistributedObject::pimpl_s::add_send_link(size_t id, const nTuple<int, 3> &shift,
                                               const toolbox::DataSet *ds)
{
    int dest_id;
    int send_tag;
//    std::tie(dest_id, send_tag, std::ignore) = GLOBAL_COMM.make_send_recv_tag(id, shift);
//
//    m_send_links_.emplace(
//            std::make_pair(id, send_link_s{send_tag, dest_id, {shift[0], shift[1], shift[2]}, ds}));


};

void DistributedObject::pimpl_s::add_recv_link(size_t id, const nTuple<int, 3> &shift, toolbox::DataSet *ds)
{
    int dest_id;
    int recv_tag;
//    std::tie(dest_id, recv_tag, std::ignore) = GLOBAL_COMM.make_send_recv_tag(id, shift);
//
//    m_recv_links_.emplace(
//            std::make_pair(id, recv_link_s{recv_tag, dest_id, {shift[0], shift[1], shift[2]}, ds}));

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

        ASSERT(item.second.data_set->data != nullptr);
        MPIDataType::create(item.second.data_set->data_type,
                            item.second.data_set->memory_space).
                swap(m_mpi_dtype_[count]);

        MPI_CALL(MPI_Isend(item.second.data_set->data.get(), 1,
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


        if (ds->memory_space.size() <= 0 || ds->data == nullptr)
        {
            MPI_Status status;

            MPI_CALL(MPI_Probe(dest_id, recv_tag, GLOBAL_COMM.comm(), &status));

            // When probe returns, the status object has the size and other
            // attributes of the incoming message. Get the size of the message
            int recv_num = 0;


            auto m_dtype = MPIDataType::create(ds->data_type);

            MPI_CALL(MPI_Get_count(&status, m_dtype.type(), &recv_num));

            if (recv_num == MPI_UNDEFINED)
            {
                THROW_EXCEPTION_RUNTIME_ERROR("Update Ghosts Particle fail");
            }

            ds->data = sp_alloc_memory(recv_num * ds->data_type.size_in_byte());

            size_type s_recv_num = static_cast<size_type>(recv_num);

            ds->memory_space = data::DataSpace(1, &s_recv_num);

//            ds->data_space = ds->memory_space;

            MPIDataType::create(ds->data_type).swap(m_mpi_dtype_[count]);

            ASSERT(ds->data.get() != nullptr);
            MPI_CALL(MPI_Irecv(ds->data.get(), recv_num,
                               m_mpi_dtype_[count].type(),
                               dest_id, recv_tag, GLOBAL_COMM.comm(),
                               &(m_mpi_requests_[count])));

        } else
        {
            ASSERT(ds->data.get() != nullptr);
            MPIDataType::create(ds->data_type, ds->memory_space).swap(m_mpi_dtype_[count]);
            MPI_CALL(MPI_Irecv(ds->data.get(), 1,
                               m_mpi_dtype_[count].type(),
                               dest_id, recv_tag, GLOBAL_COMM.comm(),
                               &(m_mpi_requests_[count])));
        }

        ++count;
    }
//
//    {
//
//        MPI_Comm mpi_global_comm = GLOBAL_COMM.comm();
//
//        MPI_Barrier(mpi_global_comm);
//
//        auto const &g_array = pool->mesh_as().global_array_;
//        if (g_array.send_recv_.size() == 0)
//        {
//            return;
//        }
//        VERBOSE << "sync ghosts (particle pool) ";
//
//        int num_of_neighbour = g_array.send_recv_.size();
//
//        MPI_Request requests[num_of_neighbour * 2];
//
//        std::vector<std::vector<value_type>> m_buffer(num_of_neighbour * 2);
//
//        int count = 0;
//
//        for (auto const &item : g_array.send_recv_)
//        {
//
//            size_t num = 0;
//            for (auto s : pool->mesh_as().select_inner(item.send_begin, item.send_end))
//            {
//                num += pool->get(s).size();
//            }
//
//            m_buffer[count].resize(num);
//
//            num = 0;
//
//            for (auto s : pool->mesh_as().select_inner(item.send_begin, item.send_end))
//            {
//                for (auto const &p : pool->get(s))
//                {
//                    m_buffer[count][num] = p;
//                    ++num;
//                }
//            }
//
//            MPI_Isend(&m_buffer[count][0], m_buffer[count].size() * sizeof(value_type),
//                      MPI_BYTE, item.dest, item.send_tag, mpi_global_comm, &requests[count]);
//            ++count;
//
//        }
//
//        for (auto const &item : g_array.send_recv_)
//        {
//            pool->remove(pool->mesh_as().select_outer(item.recv_begin, item.recv_end));
//
//            MPI_Status status;
//
//            MPI_Probe(item.dest, item.recv_tag, mpi_global_comm, &status);
//
//            // When probe returns, the status object has the size and other
//            // attributes of the incoming message. Get the size of the message
//            int mem_size = 0;
//            MPI_Get_count(&status, MPI_BYTE, &mem_size);
//
//            if (mem_size == MPI_UNDEFINED)
//            {
//                RUNTIME_ERROR("Update Ghosts particle fail");
//            }
//            m_buffer[count].resize(mem_size / sizeof(value_type));
//
//            MPI_Irecv(&m_buffer[count][0], m_buffer[count].size() * sizeof(value_type),
//                      MPI_BYTE, item.dest, item.recv_tag, mpi_global_comm, &requests[count]);
//            ++count;
//        }
//
//        MPI_Waitall(num_of_neighbour, requests, MPI_STATUSES_IGNORE);
//
//        auto cell_buffer = pool->create_child();
//        for (int i = 0; i < num_of_neighbour; ++i)
//        {
//            typename manifold_type::coordinate_tuple xmin, xmax, extents;
//
//            std::tie(xmin, xmax) = pool->mesh_as().get_extents();
//
//            bool id = true;
//            for (int n = 0; n < 3; ++n)
//            {
//                if (g_array.send_recv_[i].recv_begin[n]
//                    < pool->mesh_as().global_begin_[n])
//                {
//                    extents[n] = xmin[n] - xmax[n];
//                }
//                else if (g_array.send_recv_[i].recv_begin[n]
//                         >= pool->mesh_as().global_end_[n])
//                {
//                    extents[n] = xmax[n] - xmin[n];
//                }
//                else
//                {
//                    extents[n] = 0;
//                }
//
//            }
//
//            if (extents[0] != 0.0 || extents[1] != 0.0 || extents[2] != 0.0)
//            {
//                for (auto p : m_buffer[num_of_neighbour + i])
//                {
//                    p.x += extents;
//                    cell_buffer.push_back(std::move(p));
//                }
//            }
//            else
//            {
//                std::copy(m_buffer[num_of_neighbour + i].begin(),
//                          m_buffer[num_of_neighbour + i].end(),
//                          std::back_inserter(cell_buffer));
//            }
//
//        }
//
//        MPI_Barrier(mpi_global_comm);
//
//        pool->add(&cell_buffer);
//
//    }
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
    if (m_mpi_requests_.size() > 0)
    {
        int flag = 0;

        int count = static_cast<int>(m_mpi_requests_.size());

        MPI_Request *array_of_requests = &(const_cast<pimpl_s *>(this)->m_mpi_requests_[0]);

        MPI_CALL(MPI_Testall(count, array_of_requests, &flag, MPI_STATUSES_IGNORE));

        return flag != 0;
    }

    return true;

}


//! Default constructor
DistributedObject::DistributedObject() : pimpl_(new pimpl_s()) {}

DistributedObject::~DistributedObject() {}

void DistributedObject::clear() { pimpl_->clear(); }

void DistributedObject::sync() { pimpl_->sync(); }

void DistributedObject::wait() { pimpl_->wait(); }

bool DistributedObject::is_ready() const { return pimpl_->is_ready(); }


void DistributedObject::add_send_link(size_t id, const nTuple<int, 3> &offset, const toolbox::DataSet *ds)
{
    return pimpl_->add_send_link(id, offset, ds);

};

void DistributedObject::add_recv_link(size_t id, const nTuple<int, 3> &offset, toolbox::DataSet *ds)
{
    return pimpl_->add_recv_link(id, offset, ds);
}


}}//namespace simpla{ namespace parallel