/**
 * @file rect_mesh.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_RECT_MESH_H
#define SIMPLA_RECT_MESH_H

#include "../../parallel/mpi_update.h"
#include "../mesh_ids.h"
namespace simpla
{
namespace policy
{

class RectMesh : public MeshIDs_<4>
{
public:

	static constexpr size_t ndims = 3;


	typedef MeshIDs_<4> mids;

	using mids::index_tuple;
	using mids::range_type;


private:

	bool m_is_valid_ = false;


public:
	constexpr bool is_valid() const
	{
		return m_is_valid_;
	}

	void swap(RectMesh &other)
	{

	}


	typedef typename mids::id_type id_type;

	typedef typename mids::id_tuple id_tuple;

	typedef typename mids::index_type index_type;


/**
 *
 *   -----------------------------5
 *   |                            |
 *   |     ---------------4       |
 *   |     |              |       |
 *   |     |  ********3   |       |
 *   |     |  *       *   |       |
 *   |     |  *       *   |       |
 *   |     |  *       *   |       |
 *   |     |  2********   |       |
 *   |     1---------------       |
 *   0-----------------------------
 *
 *	5-0 = dimensions
 *	4-1 = e-d = ghosts
 *	2-1 = counts
 *
 *	0 = id_begin
 *	5 = id_end
 *
 *	1 = id_local_outer_begin
 *	4 = id_local_outer_end
 *
 *	2 = id_local_inner_begin
 *	3 = id_local_inner_end
 *
 *
 */

	id_type m_id_min_;

	id_type m_id_max_;

	id_type m_id_local_min_;

	id_type m_id_local_max_;

	id_type m_id_memory_min_;

	id_type m_id_memory_max_;


public:


	template<typename TI>
	void dimensions(TI const &d)
	{

		m_id_min_ = mids::ID_ZERO;
		m_id_max_ = m_id_min_ + mids::pack_index(d);

	}

	index_tuple dimensions() const { return mids::unpack_index(m_id_max_ - m_id_min_); }

	template<size_t IFORM>
	size_t max_hash() const { return mids::template max_hash<IFORM>(m_id_memory_min_, m_id_memory_max_); }

	size_t hash(id_type s) const { return mids::hash(s, m_id_memory_min_, m_id_memory_max_); }

	template<size_t IFORM>
	id_type pack_relative_index(index_type i, index_type j,
			index_type k, index_type n = 0) const
	{
		return mids::pack_index(nTuple<index_type, 3>({i, j, k}), mids::template sub_index_to_id<IFORM>(n)) + m_id_min_;
	}

	nTuple<index_type, ndims + 1> unpack_relative_index(id_type s) const
	{
		nTuple<index_type, ndims + 1> res;
		res = mids::unpack_index(s - m_id_min_);
		res[ndims] = mids::sub_index(s);
		return std::move(res);
	}


	std::tuple<id_tuple, id_tuple> index_box() const
	{
		return std::make_tuple(mids::unpack_index(m_id_min_),
				mids::unpack_index(m_id_max_));
	}

	std::tuple<id_tuple, id_tuple> local_index_box() const
	{
		return std::make_tuple(mids::unpack_index(m_id_local_min_),
				mids::unpack_index(m_id_local_max_));
	}


	auto box() const DECL_RET_TYPE(std::forward_as_tuple(m_id_local_min_, m_id_local_max_))

	template<typename T>
	bool in_box(T const &x) const
	{
		auto b = mids::unpack_index(m_id_local_min_);
		auto e = mids::unpack_index(m_id_local_max_);
		return (b[1] <= x[1]) && (b[2] <= x[2]) && (b[0] <= x[0])  //
				&& (e[1] > x[1]) && (e[2] > x[2]) && (e[0] > x[0]);

	}

	bool in_box(id_type s) const
	{
		return in_box(mids::unpack_index(s));
	}


	template<typename ...Args>
	void reset(Args &&...args)
	{
		RectMesh(mids::pack(args)...).swap(*this);
	}


	//! @ingroup interface
	//! @{

	void deploy(size_t const *gw = nullptr);


	range_type range(int nid = 0) const
	{
		return range_type(m_id_local_min_, m_id_local_max_, nid);
	}

	template<int I> range_type make_range() const
	{
		return range_type(m_id_local_min_, m_id_local_max_, mids::sub_index_to_id<I>());
	}


	template<int IFORM> DataSpace dataspace() const;

	template<int IFORM> void ghost_shape(std::vector<mpi_ghosts_shape_s> *res) const;

	template<int IFORM> std::vector<mpi_ghosts_shape_s> ghost_shape() const;


}; //class RectMesh


void RectMesh::deploy(size_t const *gw)
{
/**
* Decompose
*/

	if (GLOBAL_COMM.num_of_process() > 1)
	{
		auto idx_b = mids::unpack_index(m_id_min_);

		auto idx_e = mids::unpack_index(m_id_max_);

		GLOBAL_COMM.
				decompose(ndims, &idx_b[0], &idx_e[0]
		);

		typename mids::index_tuple ghost_width;

		if (gw != nullptr)
		{
			ghost_width = gw;
		}
		else
		{
//			ghost_width = mids::DEFAULT_GHOST_WIDTH;
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
//				> (mids::unpack_index(
//								m_id_max_ - m_id_min_))
//				+ " , Local dimensions=" + type_cast
//				< std::string
//				> (mids::unpack_index(
//								m_id_local_max_ - m_id_local_min_))
//				+ " , Ghost width =" + type_cast
//				< std::string > (ghost_width) +
								"]");
			}

		}

		m_id_local_min_ = mids::pack_index(idx_b);

		m_id_local_max_ = mids::pack_index(idx_e);

		m_id_memory_min_ = m_id_local_min_ - mids::pack_index(ghost_width);

		m_id_memory_max_ = m_id_local_max_ + mids::pack_index(ghost_width);


	}
	else
	{
		m_id_local_min_ = m_id_min_;

		m_id_local_max_ = m_id_max_;

		m_id_memory_min_ = m_id_local_min_;

		m_id_memory_max_ = m_id_local_max_;

	}

}


template<int IFORM> DataSpace
RectMesh::dataspace() const
{
	nTuple<index_type, ndims + 1> f_dims;
	nTuple<index_type, ndims + 1> f_offset;
	nTuple<index_type, ndims + 1> f_count;
	nTuple<index_type, ndims + 1> f_ghost_width;

	nTuple<index_type, ndims + 1> m_dims;
	nTuple<index_type, ndims + 1> m_offset;

	int f_ndims = ndims;

	f_dims = mids::unpack_index(m_id_max_ - m_id_min_);

	f_offset = mids::unpack_index(m_id_local_min_ - m_id_min_);

	f_count = mids::unpack_index(
			m_id_local_max_ - m_id_local_min_);

	m_dims = mids::unpack_index(
			m_id_memory_max_ - m_id_memory_min_);;

	m_offset = mids::unpack_index(m_id_local_min_ - m_id_min_);

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

	res.
					select_hyperslab(&f_offset[0],
					nullptr, &f_count[0], nullptr)
			.
					set_local_shape(&m_dims[0], &m_offset[0]
			);

	return
			std::move(res);

}

template<int IFORM> void
RectMesh::ghost_shape(std::vector<mpi_ghosts_shape_s> *res) const
{
	nTuple<size_t, ndims + 1> f_local_dims;
	nTuple<size_t, ndims + 1> f_local_offset;
	nTuple<size_t, ndims + 1> f_local_count;
	nTuple<size_t, ndims + 1> f_ghost_width;

	int f_ndims = ndims;

//		f_local_dims = mids::unpack_index(
//				m_id_memory_max_ - m_id_memory_min_);

	f_local_offset = mids::unpack_index(
			m_id_local_min_ - m_id_memory_min_);

	f_local_count = mids::unpack_index(
			m_id_local_max_ - m_id_local_min_);

	f_ghost_width = mids::unpack_index(
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


template<int IFORM> std::vector<mpi_ghosts_shape_s>
RectMesh::ghost_shape() const
{
	std::vector<mpi_ghosts_shape_s> res;
	ghost_shape<IFORM>(&res);
	return std::move(res);
}
}//namespace policy
}//namespace simpla
#endif //SIMPLA_RECT_MESH_H
