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


    template<typename TG, typename TOP, typename T, typename ...Args>
    void for_each(TG const &geo, TOP const &op, T *self, Args &&... args) const;

    template<typename TG, typename TOP, typename ...Args>
    void for_each(TG const &geo, TOP const &op, Args &&... args) const;


private:

    mesh_type &m_mesh_;

    MPIComm &m_mpi_comm_;

    struct connection_node
    {
        nTuple<int, 3> coord_offset;
        nTuple<size_t, 3> send_min;
        nTuple<size_t, 3> send_max;

        nTuple<size_t, 3> recv_min;
        nTuple<size_t, 3> recv_max;
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


//	auto idx_b = base_manifold_type::unpack_index(base_manifold_type::m_id_min_);
//
//	auto idx_e = base_manifold_type::unpack_index(base_manifold_type::m_id_max_);
//
//	m_mpi_comm_.decompose(base_manifold_type::ndims, &idx_b[0], &idx_e[0]);
//
//	for (int i = 0; i < base_manifold_type::ndims; ++i)
//	{
//		if (idx_b[i] + 1 == idx_e[i])
//		{
//			m_ghost_width_[i] = 0;
//		}
//		else if (idx_e[i] <= idx_b[i] + m_ghost_width_[i] * 2)
//		{
//			ERROR("Dimension is to small to split!["
////				" Dimensions= " + type_cast < std::string
////				> (base_manifold_type::unpack_index(
////								m_id_max_ - m_id_min_))
////				+ " , Local dimensions=" + type_cast
////				< std::string
////				> (base_manifold_type::unpack_index(
////								m_id_local_max_ - m_id_local_min_))
////				+ " , Ghost width =" + type_cast
////				< std::string > (ghost_width) +
//					"]");
//		}
//
//	}
//
//	base_manifold_type::m_id_local_min_ = base_manifold_type::pack_index(idx_b);
//
//	base_manifold_type::m_id_local_max_ = base_manifold_type::pack_index(idx_e);
//
//	base_manifold_type::m_id_memory_min_ = base_manifold_type::m_id_local_min_ - base_manifold_type::pack_index(m_ghost_width_);
//
//	base_manifold_type::m_id_memory_max_ = base_manifold_type::m_id_local_max_ + base_manifold_type::pack_index(m_ghost_width_);
//
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> l_dims, l_offset, l_stride, l_count, l_block;
//
//	l_dims = local_shape.dimensions;
//	l_offset = local_shape.offset;
//	l_stride = local_shape.stride;
//	l_count = local_shape.count;
//	l_block = local_shape.block;
//
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;
//
//	for (unsigned int tag = 0, tag_e = (1U << (base_manifold_type::ndims * 2)); tag < tag_e; ++tag)
//	{
//		nTuple<int, 3> coord_shift;
//
//		bool tag_is_valid = true;
//
//		for (int n = 0; n < base_manifold_type::ndims; ++n)
//		{
//			if (((tag >> (n * 2)) & 3UL) == 3UL)
//			{
//				tag_is_valid = false;
//				break;
//			}
//
//			coord_shift[n] = ((tag >> (n * 2)) & 3U) - 1;
//
//			switch (coord_shift[n])
//			{
//			case 0:
//				send_count[n] = l_count[n];
//				send_offset[n] = l_offset[n];
//				recv_count[n] = l_count[n];
//				recv_offset[n] = l_offset[n];
//				break;
//			case -1: //left
//
//				send_count[n] = m_ghost_width_[n];
//				send_offset[n] = l_offset[n];
//
//				recv_count[n] = m_ghost_width_[n];
//				recv_offset[n] = l_offset[n] - m_ghost_width_[n];
//
//				break;
//			case 1: //right
//				send_count[n] = m_ghost_width_[n];
//				send_offset[n] = l_offset[n] + l_count[n] - m_ghost_width_[n];
//
//				recv_count[n] = m_ghost_width_[n];
//				recv_offset[n] = l_offset[n] + l_count[n];
//				break;
//			default:
//				tag_is_valid = false;
//				break;
//			}
//
//			if (send_count[n] == 0 || recv_count[n] == 0)
//			{
//				tag_is_valid = false;
//				break;
//			}
//
//		}
//
//		if (tag_is_valid && (coord_shift[0] != 0 || coord_shift[1] != 0 || coord_shift[2] != 0))
//		{
//			add_link(
//					&coord_shift[0],
//					m_mesh_.local_range(send_offset, send_count),
//					m_mesh_.local_range(recv_offset, recv_count)
//			);
//
}

template<typename TMesh>
template<typename TG, typename TOP, typename T, typename ...Args>
void ParallelPolicy<TMesh>::for_each(TG const &geo, TOP const &op, T *self, Args &&... args) const
{
    static constexpr int IFORM = traits::iform<T>::value;
    typedef traits::value_type_t<T> value_type;
    auto r = m_mesh_.template range<IFORM>();
    typedef decltype(r) range_type;

    for (auto const &item:m_boundary_box_)
    {
        parallel::parallel_for(m_mesh_.template make_range<IFORM>(std::get<0>(item), std::get<1>(item)),
                               [&](range_type const &r)
                               {
                                   for (auto const &s:r)
                                   {
                                       op(geo.access(*self, s), geo.access(std::forward<Args>(args), s)...);
                                   }
                               });

    }


    //    DistributedObject dist_obj(m_mpi_comm_);
    // TODO create distributed object
    //
    //    for (auto const &item:m_connections_)
    //    {
    //        dist_obj.add_link();
    //    }
    //    dist_obj.sync();

    parallel::parallel_for(m_mesh_.template make_range<IFORM>(std::get<0>(m_center_box_), std::get<1>(m_center_box_)),
                           [&](range_type const &r)
                           {
                               for (auto const &s:r)
                               {
                                   op(geo.access(*self, s), geo.access(std::forward<Args>(args), s)...);
                               }
                           });


    //TODO wait
//    dist_obj.wait();

};

template<typename TMesh>
template<typename TG, typename TOP, typename ...Args>
void ParallelPolicy<TMesh>::for_each(TG const &geo, TOP const &op, Args &&... args) const
{
    static constexpr int IFORM = traits::iform<traits::unpack_type<0, Args>::type>::value;
    typedef traits::value_type_t<traits::unpack_type<0, Args>::type> value_type;
    typedef decltype(m_mesh_.template make_range<IFORM>()) range_type;

    parallel::parallel_for(m_mesh_.template make_range<IFORM>(),
                           [&](range_type const &r)
                           {
                               for (auto const &s:r)
                               {
                                   op(geo.access(std::forward<Args>(args), s)...);
                               }
                           });


};
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
