//
// Created by salmon on 16-6-6.
//
#include "MeshAttribute.h"

#ifdef HAS_MPI
#   include "../parallel/DistributedObject.h"
#endif
namespace simpla { namespace mesh
{
struct MeshAttribute::pimpl_s
{
    parallel::DistributedObject m_dist_obj_;

    struct link_s
    {
        bool is_presient = false;
        MeshEntityRange m_send_range_;
        MeshEntityRange m_recv_range_;
        data_model::DataSet m_recv_dataset_;
        data_model::DataSet m_send_dataset_;
    };
    std::map<int, link_s> m_links_;


    MeshBase const *m_mesh_;
    std::shared_ptr<void> m_data_;
    std::shared_ptr<MeshAttribute> m_holder_;


};

MeshAttribute::MeshAttribute() : m_pimpl_(new pimpl_s) { }

MeshAttribute::MeshAttribute(MeshBase const *m, MeshAttribute *other) : m_pimpl_(new pimpl_s)
{
    m_pimpl_->m_mesh_ = m;
    m_pimpl_->m_data_ = nullptr;
    m_pimpl_->m_holder_ = std::shared_ptr<MeshAttribute>(other);
}

MeshAttribute::MeshAttribute(std::shared_ptr<MeshAttribute> other) : m_pimpl_(new pimpl_s)
{
    assert (other->is_valid());

    m_pimpl_->m_mesh_ = other->m_pimpl_->m_mesh_;
    m_pimpl_->m_data_ = other->m_pimpl_->m_data_;
    m_pimpl_->m_holder_ = other;

}

MeshAttribute::MeshAttribute(MeshAttribute const &other)
{
    assert (other.is_valid());

    std::unique_ptr<pimpl_s>(new pimpl_s).swap(m_pimpl_);
    m_pimpl_->m_mesh_ = other.m_pimpl_->m_mesh_;
    m_pimpl_->m_data_ = other.m_pimpl_->m_data_;
    m_pimpl_->m_holder_ = other.m_pimpl_->m_holder_;

}


MeshAttribute::~MeshAttribute() { }

bool MeshAttribute::is_valid() const { return m_pimpl_ != nullptr; }

bool MeshAttribute::empty() const { return (!is_valid()) || (m_pimpl_->m_data_ == nullptr); }

void MeshAttribute::swap(MeshAttribute &other)
{
    std::swap(m_pimpl_, other.m_pimpl_);
}

std::shared_ptr<MeshAttribute>
MeshAttribute::holder() { return m_pimpl_->m_holder_; }

MeshBase const *
MeshAttribute::mesh() const { return m_pimpl_->m_mesh_; }

void *MeshAttribute::data() { return m_pimpl_->m_data_.get(); }

const void *MeshAttribute::data() const { return m_pimpl_->m_data_.get(); }

size_type MeshAttribute::size_in_byte() const
{
    assert(m_pimpl_ != nullptr);
    assert(m_pimpl_->m_mesh_ != nullptr);
    return m_pimpl_->m_mesh_->max_hash(entity_type()) * entity_size_in_byte();
}

bool MeshAttribute::deploy()
{
    assert(m_pimpl_ != nullptr);

    if (m_pimpl_->m_data_ == 0x0)
    {
        m_pimpl_->m_data_ = sp_alloc_memory(size_in_byte());
    }
    return true;
}

void MeshAttribute::clear()
{
    deploy();
    memset(m_pimpl_->m_data_.get(), 0, size_in_byte());
}

void MeshAttribute::sync(bool is_blocking)
{
#ifdef HAS_MPI
    //
    //    typedef typename data_model::DataSpace::index_tuple index_tuple;
    //
    //    int ndims;
    //    index_tuple dimensions;
    //    index_tuple start;
    ////    index_tuple stride;
    //    index_tuple count;
    ////    index_tuple block;
    //
    //
    //
    //    std::tie(ndims, dimensions, start, std::ignore, count, std::ignore)
    //            = ds->memory_space.shape();
    //
    //
    //    ASSERT(start + count <= dimensions);
    //
    //    index_tuple ghost_width = start;
    //
    //
    //    nTuple<ptrdiff_t, 3> send_offset;
    //    nTuple<size_t, 3> send_count;
    //    nTuple<ptrdiff_t, 3> recv_offset;
    //    nTuple<size_t, 3> recv_count;
    //
    //    for (unsigned int tag = 0, tag_e = (1U << (ndims * 2)); tag < tag_e; ++tag)
    //    {
    //        nTuple<int, 3> coord_offset;
    //
    //        bool tag_is_valid = true;
    //
    //        for (int n = 0; n < ndims; ++n)
    //        {
    //            if (((tag >> (n * 2)) & 3UL) == 3UL)
    //            {
    //                tag_is_valid = false;
    //                break;
    //            }
    //
    //            coord_offset[n] = ((tag >> (n * 2)) & 3U) - 1;
    //
    //            switch (coord_offset[n])
    //            {
    //                case 0:
    //                    send_offset[n] = start[n];
    //                    send_count[n] = count[n];
    //                    recv_offset[n] = start[n];
    //                    recv_count[n] = count[n];
    //
    //                    break;
    //                case -1: //left
    //
    //                    send_offset[n] = start[n];
    //                    send_count[n] = ghost_width[n];
    //                    recv_offset[n] = start[n] - ghost_width[n];
    //                    recv_count[n] = ghost_width[n];
    //
    //
    //                    break;
    //                case 1: //right
    //                    send_offset[n] = start[n] + count[n] - ghost_width[n];
    //                    send_count[n] = ghost_width[n];
    //                    recv_offset[n] = start[n] + count[n];
    //                    recv_count[n] = ghost_width[n];
    //
    //                    break;
    //                default:
    //                    tag_is_valid = false;
    //                    break;
    //            }
    //
    //            if (send_count[n] == 0 || recv_count[n] == 0)
    //            {
    //                tag_is_valid = false;
    //                break;
    //            }
    //
    //        }
    //
    //        if (tag_is_valid && (coord_offset[0] != 0 || coord_offset[1] != 0 || coord_offset[2] != 0))
    //        {
    //            try
    //            {
    //
    //                data_model::DataSet send_ds(*ds);
    //
    //                data_model::DataSet recv_ds(*ds);
    //
    //                send_ds.memory_space.select_hyperslab(&send_offset[0], nullptr, &send_count[0], nullptr);
    //
    //                recv_ds.memory_space.select_hyperslab(&recv_offset[0], nullptr, &recv_count[0], nullptr);
    //
    //
    //                m_pimpl_->m_dist_obj_.add_send_link(id, send_offset, std::move(send_ds));
    //
    //                m_pimpl_->m_dist_obj_.add_send_link(id, recv_offset, std::move(recv_ds));
    //
    //
    //            }
    //            catch (std::exception const &error)
    //            {
    //                RUNTIME_ERROR << "add communication link error" << error.what() << std::endl;
    //
    //            }
    //        }
    //
    //    }


        m_pimpl_->m_dist_obj_.sync();
        if (is_blocking) { wait(); }
#endif
}

void MeshAttribute::wait()
{
#ifdef HAS_MPI
    LOG_CMD_DESC(" SYNC [" + get_class_name() + "]", m_pimpl_->m_dist_obj_.wait());
#endif
}

bool MeshAttribute::is_ready() const
{
    return m_pimpl_->m_dist_obj_.is_ready();
}
}}//namespace simpla{namespace get_mesh{