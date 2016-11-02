//
// Created by salmon on 16-6-6.
//
#include "DataEntity.h"


namespace simpla { namespace data
{




//
//void DataEntityHeavy::sync(bool is_blocking)
//{
//#ifdef HAS_MPI
//    //
//    //    typedef typename data_model::DataSpace::index_tuple index_tuple;
//    //
//    //    int m_ndims_;
//    //    index_tuple dimensions;
//    //    index_tuple start;
//    ////    index_tuple stride;
//    //    index_tuple count;
//    ////    index_tuple block;
//    //
//    //
//    //
//    //    std::tie(m_ndims_, dimensions, start, std::ignore, count, std::ignore)
//    //            = ds->memory_space.m_global_dims_();
//    //
//    //
//    //    ASSERT(start + count <= dimensions);
//    //
//    //    index_tuple m_ghost_width_ = start;
//    //
//    //
//    //    nTuple<ptrdiff_t, 3> send_offset;
//    //    nTuple<size_t, 3> send_count;
//    //    nTuple<ptrdiff_t, 3> recv_offset;
//    //    nTuple<size_t, 3> recv_count;
//    //
//    //    for (unsigned int tag = 0, tag_e = (1U << (m_ndims_ * 2)); tag < tag_e; ++tag)
//    //    {
//    //        nTuple<int, 3> coord_offset;
//    //
//    //        bool tag_is_valid = true;
//    //
//    //        for (int n = 0; n < m_ndims_; ++n)
//    //        {
//    //            if (((tag >> (n * 2)) & 3UL) == 3UL)
//    //            {
//    //                tag_is_valid = false;
//    //                break;
//    //            }
//    //
//    //            coord_offset[n] = ((tag >> (n * 2)) & 3U) - 1;
//    //
//    //            switch (coord_offset[n])
//    //            {
//    //                case 0:
//    //                    send_offset[n] = start[n];
//    //                    send_count[n] = count[n];
//    //                    recv_offset[n] = start[n];
//    //                    recv_count[n] = count[n];
//    //
//    //                    break;
//    //                case -1: //left
//    //
//    //                    send_offset[n] = start[n];
//    //                    send_count[n] = m_ghost_width_[n];
//    //                    recv_offset[n] = start[n] - m_ghost_width_[n];
//    //                    recv_count[n] = m_ghost_width_[n];
//    //
//    //
//    //                    break;
//    //                case 1: //right
//    //                    send_offset[n] = start[n] + count[n] - m_ghost_width_[n];
//    //                    send_count[n] = m_ghost_width_[n];
//    //                    recv_offset[n] = start[n] + count[n];
//    //                    recv_count[n] = m_ghost_width_[n];
//    //
//    //                    break;
//    //                default:
//    //                    tag_is_valid = false;
//    //                    break;
//    //            }
//    //
//    //            if (send_count[n] == 0 || recv_count[n] == 0)
//    //            {
//    //                tag_is_valid = false;
//    //                break;
//    //            }
//    //
//    //        }
//    //
//    //        if (tag_is_valid && (coord_offset[0] != 0 || coord_offset[1] != 0 || coord_offset[2] != 0))
//    //        {
//    //            try
//    //            {
//    //
//    //                data_model::DataSet send_ds(*ds);
//    //
//    //                data_model::DataSet recv_ds(*ds);
//    //
//    //                send_ds.memory_space.select_hyperslab(&send_offset[0], nullptr, &send_count[0], nullptr);
//    //
//    //                recv_ds.memory_space.select_hyperslab(&recv_offset[0], nullptr, &recv_count[0], nullptr);
//    //
//    //
//    //                m_self_->m_dist_obj_.add_send_link(id, send_offset, std::move(send_ds));
//    //
//    //                m_self_->m_dist_obj_.add_send_link(id, recv_offset, std::move(recv_ds));
//    //
//    //
//    //            }
//    //            catch (std::exception const &error)
//    //            {
//    //                RUNTIME_ERROR << "add communication link error" << error.what() << std::endl;
//    //
//    //            }
//    //        }
//    //
//    //    }
//
//
//        m_self_->m_dist_obj_.sync();
//        if (is_blocking) { wait(); }
//#endif
//}

}}//namespace simpla{namespace get_mesh{