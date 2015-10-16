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
template<typename ...> struct ParallelPolicy;

template<typename TGeo>
struct ParallelPolicy<TGeo>
{

private:
	typedef TGeo geometry_type;

	enum { DEFAULT_GHOST_WIDTH = 2 };

	typedef ParallelPolicy<geometry_type> this_type;

	geometry_type const &m_geo_;

public:
	ParallelPolicy(geometry_type &geo) : m_geo_(geo)
	{
	}

	virtual ~ParallelPolicy()
	{
	}

	template<typename TDict> void load(TDict const &) { }


	template<typename OS> OS &print(OS &os) const
	{
		os << "\t ParallelPolicy={ Default }," << std::endl;
		return os;
	}

	void decompose(size_t const *gw = nullptr);


}; //template<typename TGeo> struct ParallelPolicy

template<typename TGeo>
void ParallelPolicy<TGeo>::decompose(size_t const *gw)
{


	if (GLOBAL_COMM.num_of_process() <= 1)
	{
		return;
	}


	auto idx_b = geometry_type::unpack_index(geometry_type::m_id_min_);

	auto idx_e = geometry_type::unpack_index(geometry_type::m_id_max_);

	GLOBAL_COMM.decompose(geometry_type::ndims, &idx_b[0], &idx_e[0]);

	typename geometry_type::index_tuple ghost_width;

	if (gw != nullptr)
	{
		ghost_width = gw;
	}
	else
	{
		ghost_width = DEFAULT_GHOST_WIDTH;
	}

	for (int i = 0; i < geometry_type::ndims; ++i)
	{

		if (idx_b[i] + 1 == idx_e[i])
		{
			ghost_width[i] = 0;
		}
		else if (idx_e[i] <= idx_b[i] + ghost_width[i] * 2)
		{
			ERROR("Dimension is to small to split!["
//				" Dimensions= " + type_cast < std::string
//				> (geometry_type::unpack_index(
//								m_id_max_ - m_id_min_))
//				+ " , Local dimensions=" + type_cast
//				< std::string
//				> (geometry_type::unpack_index(
//								m_id_local_max_ - m_id_local_min_))
//				+ " , Ghost width =" + type_cast
//				< std::string > (ghost_width) +
					"]");
		}

	}

	geometry_type::m_id_local_min_ = geometry_type::pack_index(idx_b);

	geometry_type::m_id_local_max_ = geometry_type::pack_index(idx_e);

	geometry_type::m_id_memory_min_ = geometry_type::m_id_local_min_ - geometry_type::pack_index(ghost_width);

	geometry_type::m_id_memory_max_ = geometry_type::m_id_local_max_ + geometry_type::pack_index(ghost_width);



//	template<size_t IFORM>
//	void ghost_shape(
//			std::vector<mpi_ghosts_shape_s> *res) const
//	{
//		nTuple<size_t, ndims + 1> f_local_dims;
//		nTuple<size_t, ndims + 1> f_local_offset;
//		nTuple<size_t, ndims + 1> f_local_count;
//		nTuple<size_t, ndims + 1> f_ghost_width;
//
//		int f_ndims = ndims;
//
////		f_local_dims = geometry_type::unpack_index(
////				m_id_memory_max_ - m_id_memory_min_);
//
//		f_local_offset = geometry_type::unpack_index(
//				m_id_local_min_ - m_id_memory_min_);
//
//		f_local_count = geometry_type::unpack_index(
//				m_id_local_max_ - m_id_local_min_);
//
//		f_ghost_width = geometry_type::unpack_index(
//				m_id_local_min_ - m_id_memory_min_);
//
//		if ((IFORM == EDGE || IFORM == FACE))
//		{
//			f_ndims = ndims + 1;
//			f_local_offset[ndims] = 0;
//			f_local_count[ndims] = 3;
//			f_ghost_width[ndims] = 0;
//		}
//		else
//		{
//			f_ndims = ndims;
//
////			f_local_dims[ndims] = 1;
//			f_local_offset[ndims] = 0;
//			f_local_count[ndims] = 1;
//			f_ghost_width[ndims] = 0;
//
//		}
//
//		get_ghost_shape(f_ndims, &f_local_offset[0], nullptr, &f_local_count[0],
//				nullptr, &f_ghost_width[0], res);
//
//	}

}


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
