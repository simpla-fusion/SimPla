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
struct ParallelPolicy
{
	typedef TGeo geometry_type;


	typedef ParallelPolicy<geometry_type> this_type;

	geometry_type const &m_geo_;
public:
	ParallelPolicy(geometry_type &geo) : m_geo_(geo)
	{
	}

	virtual ~ParallelPolicy()
	{
	}


	template<size_t IFORM>
	void ghost_shape(
			std::vector<mpi_ghosts_shape_s> *res) const
	{
		nTuple<size_t, ndims + 1> f_local_dims;
		nTuple<size_t, ndims + 1> f_local_offset;
		nTuple<size_t, ndims + 1> f_local_count;
		nTuple<size_t, ndims + 1> f_ghost_width;

		int f_ndims = ndims;

//		f_local_dims = m::unpack_index(
//				m_id_memory_max_ - m_id_memory_min_);

		f_local_offset = m::unpack_index(
				m_id_local_min_ - m_id_memory_min_);

		f_local_count = m::unpack_index(
				m_id_local_max_ - m_id_local_min_);

		f_ghost_width = m::unpack_index(
				m_id_local_min_ - m_id_memory_min_);

		if ((IFORM == EDGE || IFORM == FACE))
		{
			f_ndims = ndims + 1;
			f_local_offset[ndims] = 0;
			f_local_count[ndims] = 3;
			f_ghost_width[ndims] = 0;
		}
		else
		{
			f_ndims = ndims;

//			f_local_dims[ndims] = 1;
			f_local_offset[ndims] = 0;
			f_local_count[ndims] = 1;
			f_ghost_width[ndims] = 0;

		}

		get_ghost_shape(f_ndims, &f_local_offset[0], nullptr, &f_local_count[0],
				nullptr, &f_ghost_width[0], res);

	}

	template<size_t IFORM>
	std::vector<mpi_ghosts_shape_s> ghost_shape() const
	{
		std::vector<mpi_ghosts_shape_s> res;
		ghost_shape<IFORM>(&res);
		return std::move(res);
	}
}; //template<typename TGeo> struct ParallelPolicy
}//namespace simpla
#endif //SIMPLA_PARALLEL_H
