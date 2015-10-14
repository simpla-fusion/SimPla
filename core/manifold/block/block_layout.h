/**
 *
 * @file block_layout.h
 * Created by salmon on 7/6/15.
 */
#ifndef SIMPLA_MESH_BLOCK_BLOCK_LAYOUT_H
#define SIMPLA_MESH_BLOCK_BLOCK_LAYOUT_H

#include <list>
#include "../../parallel/mpi_comm.h"
#include "../../parallel/mpi_update.h"
#include "../../gtl/dataset/dataspace.h"
#include "block.h"


namespace simpla
{
namespace mesh
{


template<int NDIMS> struct BlockConnection;

/**
 *   Layout<Block> represents a set of Blocks in the same index space
 */

template<int NDIMS = 3>
struct BlockLayout
{
	static constexpr int ndims = NDIMS;


	typedef Block<NDIMS> block_type;

	typedef BlockLayout<NDIMS> layout_type;


	typedef BlockConnection<NDIMS> connection_type;

	typedef typename block_type::id_type id_type;

	typedef typename block_type::id_tuple id_tuple;

	typedef typename block_type::index_type index_type;

	typedef typename block_type::index_tuple index_tuple;

	typedef size_t id_tag_type;


	int m_level_ = 4;


	int m_proc_rank_ = 0;


	int level() const
	{
		return m_level_;
	}

	int proc_rank() const
	{
		return m_proc_rank_;
	}


	std::map<id_tag_type, layout_type> m_finer_layout_;

	/**
	 *
	 *   -----------------------------5
	 *   |                            |
	 *   |     ---------------4       |
	 *   |     |       B      |       |
	 *   |     |  ********3---|       |
	 *   |     |  *     * *   |       |
	 *   |     |  *   A *a* b |       |
	 *   |     |  *     * *   |       |
	 *   |     |  2********---|       |
	 *   |     1---------------       |
	 *   0-----------------------------
	 *
	 *  A is inner box;
	 *  B is outer box;
	 *  B - A is ghost region
	 *
	 *  a is send box
	 *  b is recv box
	 *
	 *	5-0 = dimensions
	 *	4-1 = e-d = ghosts
	 *	2-1 = counts
	 *
	  *
	 *
	 */
	block_type m_inner_box_;

	block_type m_outer_box_;


	const block_type &inner_box() const
	{
		return m_inner_box_;
	}

	const block_type &outer_box() const
	{
		return m_outer_box_;
	}


	block_id add(index_tuple const &dimension)
	{

	}


	block_id split(index_tuple const &dim)
	{

	}

	/**
	 *  Return the minimal Block that contains all siblings.
	 */
	block_type universe() const
	{

	}

	void deploy(size_t const *gw = nullptr)
	{
/**
* Decompose
*/

		if (GLOBAL_COMM.num_of_process() > 1)
		{
			auto idx_b = m::unpack_index(m_id_min_);

			auto idx_e = m::unpack_index(m_id_max_);

			GLOBAL_COMM.decompose(ndims, &idx_b[0], &idx_e[0]);

			typename m::index_tuple ghost_width;

			if (gw != nullptr)
			{
				ghost_width = gw;
			}
			else
			{
				ghost_width = DEFAULT_GHOST_WIDTH;
			}

			for (
					int i = 0;
					i < ndims;
					++i)
			{

				if (idx_b[i] + 1 == idx_e[i])
				{
					ghost_width[i] = 0;
				}
				else if (idx_e[i] <= idx_b[i] + ghost_width[i] * 2)
				{
					ERROR(
							"Dimension is to small to split!["
//				" Dimensions= " + type_cast < std::string
//				> (m::unpack_index(
//								m_id_max_ - m_id_min_))
//				+ " , Local dimensions=" + type_cast
//				< std::string
//				> (m::unpack_index(
//								m_id_local_max_ - m_id_local_min_))
//				+ " , Ghost width =" + type_cast
//				< std::string > (ghost_width) +
									"]");
				}

			}

			m_id_local_min_ = m::pack_index(idx_b);

			m_id_local_max_ = m::pack_index(idx_e);

			m_id_memory_min_ = m_id_local_min_ - m::pack_index(ghost_width);

			m_id_memory_max_ = m_id_local_max_ + m::pack_index(ghost_width);


		}
		else
		{
			m_id_local_min_ = m_id_min_;

			m_id_local_max_ = m_id_max_;

			m_id_memory_min_ = m_id_local_min_;

			m_id_memory_max_ = m_id_local_max_;

		}

	}

	template<size_t IFORM, int LEVEL>
	DataSpace dataspace() const
	{
		nTuple<index_type, ndims + 1> f_dims;
		nTuple<index_type, ndims + 1> f_offset;
		nTuple<index_type, ndims + 1> f_count;
		nTuple<index_type, ndims + 1> f_ghost_width;

		nTuple<index_type, ndims + 1> m_dims;
		nTuple<index_type, ndims + 1> m_offset;

		int f_ndims = ndims;

		f_dims = m::unpack_index(m_id_max_ - m_id_min_);

		f_offset = m::unpack_index(m_id_local_min_ - m_id_min_);

		f_count = m::unpack_index(
				m_id_local_max_ - m_id_local_min_);

		m_dims = m::unpack_index(
				m_id_memory_max_ - m_id_memory_min_);;

		m_offset = m::unpack_index(m_id_local_min_ - m_id_min_);

		if ((IFORM == EDGE || IFORM == FACE))
		{
			f_ndims = ndims + 1;
			f_dims[ndims] = 3;
			f_offset[ndims] = 0;
			f_count[ndims] = 3;
			m_dims[ndims] = 3;
			m_offset[ndims] = 0;
		}
		else
		{
			f_ndims = ndims;
			f_dims[ndims] = 1;
			f_offset[ndims] = 0;
			f_count[ndims] = 1;
			m_dims[ndims] = 1;
			m_offset[ndims] = 0;
		}

		DataSpace res(f_ndims, &(f_dims[0]));

		res.select_hyperslab(&f_offset[0], nullptr, &f_count[0], nullptr)
				.set_local_shape(&m_dims[0], &m_offset[0]);

		return std::move(res);

	}

	template<size_t IFORM>
	void ghost_shape(std::vector<mpi_ghosts_shape_s> *res) const
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
};
}// namespace manifold
}// namespace simpla
#endif //SIMPLA_MESH_BLOCK_BLOCK_LAYOUT_H
