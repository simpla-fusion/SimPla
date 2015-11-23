/**
 * @file parallel_policy.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_PARALLEL_POLICY_H
#define SIMPLA_PARALLEL_POLICY_H

#include "../../parallel/mpi_comm.h"
#include "../../parallel/mpi_update.h"
#include "../../parallel/parallel.h"
#include "../../parallel/distributed_object.h"

namespace simpla
{
/**
 * @ingroup manifold
 */
namespace manifold { namespace policy
{


template<typename ...>
struct ParallelPolicy;

template<typename TMesh>
struct ParallelPolicy<TMesh>
{

private:
    typedef TMesh mesh_type;

    typedef ParallelPolicy<mesh_type> this_type;


public:

    typedef this_type parallel_policy;

    ParallelPolicy(mesh_type &geo) :
            m_mesh_(geo), m_mpi_comm_(SingletonHolder<MPIComm>::instance())
    {
    }

    virtual ~ParallelPolicy()
    {
    }

    template<typename TDict>
    void load(TDict const &dict)
    {
    }


    template<typename OS>
    OS &print(OS &os) const
    {
        os << "\t ParallelPolicy={ Default }," << std::endl;
        return os;
    }

    void deploy();


    template<int IFORM, typename T>
    parallel::DistributedObject sync(std::shared_ptr<void> *self) const;


    template<int IFORM, typename T, typename Func>
    void for_each(Func const &fun, std::shared_ptr<void> *f = nullptr) const;


private:

    mesh_type &m_mesh_;

    MPIComm &m_mpi_comm_;

    struct connection_node
    {
        nTuple<int, 3> coord_offset;
        nTuple<size_t, mesh_type::ndims> send_offset;
        nTuple<size_t, mesh_type::ndims> send_count;

        nTuple<size_t, mesh_type::ndims> recv_offset;
        nTuple<size_t, mesh_type::ndims> recv_count;
    };
    std::tuple<nTuple<size_t, 3>, nTuple<size_t, 3>> m_center_box_;
    std::vector<std::tuple<nTuple<size_t, 3>, nTuple<size_t, 3>>> m_boundary_box_;
    std::vector<connection_node> m_connections_;


}; //template<typename TMesh> struct ParallelPolicy




template<typename TMesh>
void ParallelPolicy<TMesh>::deploy()
{
    if (m_mpi_comm_.is_valid())
    {
        m_mesh_.decompose(m_mpi_comm_.topology(), m_mpi_comm_.coordinate());
    }

    m_center_box_ = m_mesh_.local_index_box();


//	auto idx_b = mesh_type::unpack_index(mesh_type::m_id_min_);
//
//	auto idx_e = mesh_type::unpack_index(mesh_type::m_id_max_);
//
//	m_mpi_comm_.decompose(mesh_type::ndims, &idx_b[0], &idx_e[0]);
//
//	for (int i = 0; i < mesh_type::ndims; ++i)
//	{
//		if (idx_b[i] + 1 == idx_e[i])
//		{
//			m_ghost_width_[i] = 0;
//		}
//		else if (idx_e[i] <= idx_b[i] + m_ghost_width_[i] * 2)
//		{
//			ERROR("Dimension is to small to split!["
////				" Dimensions= " + type_cast < std::string
////				> (mesh_type::unpack_index(
////								m_id_max_ - m_id_min_))
////				+ " , Local dimensions=" + type_cast
////				< std::string
////				> (mesh_type::unpack_index(
////								m_id_local_max_ - m_id_local_min_))
////				+ " , Ghost width =" + type_cast
////				< std::string > (ghost_width) +
//					"]");
//		}
//
//	}
//
//	mesh_type::m_id_local_min_ = mesh_type::pack_index(idx_b);
//
//	mesh_type::m_id_local_max_ = mesh_type::pack_index(idx_e);
//
//	mesh_type::m_id_memory_min_ = mesh_type::m_id_local_min_ - mesh_type::pack_index(m_ghost_width_);
//
//	mesh_type::m_id_memory_max_ = mesh_type::m_id_local_max_ + mesh_type::pack_index(m_ghost_width_);
//
//

    auto memory_box = m_mesh_.memory_index_box();

    auto local_box = m_mesh_.local_index_box();

    nTuple<size_t, mesh_type::ndims> l_count, l_offset, ghost_width;

    l_count = std::get<1>(local_box) - std::get<0>(local_box);
    l_offset = std::get<0>(local_box) - std::get<0>(memory_box);
    ghost_width = l_offset;


    nTuple<size_t, mesh_type::ndims> send_offset, send_count;
    nTuple<size_t, mesh_type::ndims> recv_offset, recv_count;

    for (unsigned int tag = 0, tag_e = (1U << (mesh_type::ndims * 2)); tag < tag_e; ++tag)
    {
        nTuple<int, 3> coord_shift;

        bool tag_is_valid = true;

        for (int n = 0; n < mesh_type::ndims; ++n)
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
                    send_offset[n] = l_offset[n];
                    send_count[n] = l_count[n];
                    recv_offset[n] = l_offset[n];
                    recv_count[n] = l_count[n];

                    break;
                case -1: //left

                    send_offset[n] = l_offset[n];
                    send_count[n] = ghost_width[n];
                    recv_offset[n] = l_offset[n] - ghost_width[n];
                    recv_count[n] = ghost_width[n];


                    break;
                case 1: //right
                    send_offset[n] = l_offset[n] + l_count[n] - ghost_width[n];
                    send_count[n] = ghost_width[n];
                    recv_offset[n] = l_offset[n] + l_count[n];
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

        if (tag_is_valid && (coord_shift[0] != 0 || coord_shift[1] != 0 || coord_shift[2] != 0))
        {
            m_connections_.push_back(
                    connection_node{
                            coord_shift,
                            send_offset, send_count,
                            recv_offset, recv_count
                    }
            );

        }
    }

}

template<typename TMesh>
template<int IFORM, typename T, typename Func>
void ParallelPolicy<TMesh>::for_each(Func const &fun, std::shared_ptr<void> *self) const
{
    if (self != nullptr)
    {


        for (auto const &item:m_boundary_box_)
        {
            parallel::parallel_for(m_mesh_.template make_range<IFORM>(
                    std::get<0>(item),
                    std::get<1>(item)), fun);

        }


//        auto dist_obj = sync<IFORM, T>(self);

        parallel::parallel_for(
                m_mesh_.template make_range<IFORM>(
                        std::get<0>(m_center_box_),
                        std::get<1>(m_center_box_)),
                fun);

//        dist_obj.wait();
    }
    else
    {
        parallel::parallel_for(m_mesh_.template range<IFORM>(), fun);
    }

};


template<typename TMesh>
template<int IFORM, typename T>
parallel::DistributedObject
ParallelPolicy<TMesh>::sync(std::shared_ptr<void> *data) const
{
    parallel::DistributedObject dist_obj(m_mpi_comm_);

    typedef T value_type;

    auto d_type = traits::datatype<value_type>::create();

    nTuple<size_t, mesh_type::ndims> dims = m_mesh_.dimensions();

    DataSpace mem_space(m_mesh_.ndims, &dims[0]);


    for (auto const &item:m_connections_)
    {
        dist_obj.add_link_send(&item.coord_offset[0],
                               DataSpace(mem_space).select_hyperslab(
                                       &item.send_offset[0], nullptr,
                                       &item.send_count[0], nullptr),
                               d_type, data);

        dist_obj.add_link_recv(&item.coord_offset[0],
                               DataSpace(mem_space).select_hyperslab(
                                       &item.recv_offset[0], nullptr,
                                       &item.recv_count[0], nullptr),
                               d_type, data);
    }

    dist_obj.sync();

    return std::move(dist_obj);
}


} //namespace policy
} //namespace manifold
namespace traits
{

template<typename TMesh>
struct type_id<manifold::policy::ParallelPolicy<TMesh>>
{
    static std::string name()
    {
        return "ParallelPolicy<" + type_id<TMesh>::name() + ">";
    }
};
}


}//namespace simpla
#endif //SIMPLA_PARALLEL_POLICY_H
