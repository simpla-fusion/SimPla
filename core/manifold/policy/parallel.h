/**
 * @file parallel.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_PARALLEL_H
#define SIMPLA_PARALLEL_H

#include "../../parallel/mpi_comm.h"
#include "../../parallel/mpi_update.h"

namespace simpla
{
template<typename ...>
struct ParallelPolicy;

template<typename TGeo>
struct ParallelPolicy<TGeo>
{

private:
    typedef TGeo geometry_type;

    typedef ParallelPolicy<geometry_type> this_type;


public:
    ParallelPolicy(geometry_type &geo) :
            m_geo_(geo), m_mpi_comm_(SingletonHolder<MPIComm>::instance())
    {
    }

    virtual ~ParallelPolicy()
    {
    }

    template<typename TDict>
    void load(TDict const &dict)
    {
//        ghost_width(dict["GhostWidth"].template as<nTuple<size_t, base_manifold_type::ndims> >());
    }


    template<typename OS>
    OS &print(OS &os) const
    {
        os << "\t ParallelPolicy={ Default }," << std::endl;
        return os;
    }

    void deploy();

//    const nTuple<size_t, base_manifold_type::ndims> &ghost_width() const { return m_ghost_width_; }

//    void ghost_width(const nTuple<size_t, base_manifold_type::ndims> &gw) { m_ghost_width_ = gw; }
//    nTuple<size_t, base_manifold_type::ndims> m_ghost_width_ = {DEFAULT_GHOST_WIDTH, DEFAULT_GHOST_WIDTH,
//                                                           DEFAULT_GHOST_WIDTH};

    void add_link(int const coord_offset[], typename geometry_type::range_type const &send_range,
                  typename geometry_type::range_type const &recv_range);


    template<int iform, typename TFun>
    void parallel_foreach(TFun const &fun) const
    {

        for (auto s:m_geo_.template range<iform>())
        {
            fun(s);
        }
    }


private:

    geometry_type &m_geo_;

    MPIComm &m_mpi_comm_;

    struct connection_node
    {
        nTuple<int, 3> coord_offset;
        typename geometry_type::range_type send_range;
        typename geometry_type::range_type recv_range;

    };


    std::vector<connection_node> m_connections_;

public:
    template<int IFORM>
    std::vector<connection_node> const &connections() const { return m_connections_; }

    MPIComm &comm() const { return m_mpi_comm_; }

}; //template<typename TGeo> struct ParallelPolicy


template<typename TGeo>
void ParallelPolicy<TGeo>::add_link(int const coord_offset[], typename geometry_type::range_type const &send_range,
                                    typename geometry_type::range_type const &recv_range)
{
    m_connections_.emplace_back(&coord_offset[0], send_range, recv_range);
};


template<typename TGeo>
void ParallelPolicy<TGeo>::deploy()
{
    if (m_mpi_comm_.is_valid())
    {
        m_geo_.decompose(m_mpi_comm_.topology(), m_mpi_comm_.coordinate());
    }
}

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
//					m_geo_.local_range(send_offset, send_count),
//					m_geo_.local_range(recv_offset, recv_count)
//			);
//
//		}


namespace traits
{

template<typename TGeo>
struct type_id<ParallelPolicy<TGeo>>
{
    static std::string name()
    {
        return "ParallelPolicy<" + type_id<TGeo>::name() + ">";
    }
};
}
}//namespace simpla
#endif //SIMPLA_PARALLEL_H
