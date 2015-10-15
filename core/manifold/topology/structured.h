//
// Created by salmon on 15-7-2.
//

#ifndef SIMPLA_TOPOLOGY_H
#define SIMPLA_TOPOLOGY_H

#include "mesh_ids.h"
#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/type_traits.h"

#include "topology.h"
#include <vector>

namespace simpla
{
template<typename...> struct Topology;

namespace topology
{


struct StructuredMesh : public MeshIDs_<4>
{
	enum { DEFAULT_GHOST_WIDTH = 2 };
	static constexpr int ndims = 3;

private:

	typedef StructuredMesh this_type;


	typedef MeshIDs_<4> m;

public:
	using m::id_type;
	using m::id_tuple;
	using m::index_type;
	typedef id_type value_type;
	typedef size_t difference_type;


	bool m_is_distributed_ = false;


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

	StructuredMesh()
	{

	}


	StructuredMesh(StructuredMesh const &other) :

			m_id_min_(other.m_id_min_),

			m_id_max_(other.m_id_max_),

			m_id_local_min_(other.m_id_local_min_),

			m_id_local_max_(other.m_id_local_max_),

			m_id_memory_max_(other.m_id_memory_max_),

			m_id_memory_min_(other.m_id_memory_min_)
	{

	}

	~StructuredMesh()
	{

	}

	this_type &operator=(this_type const &other)
	{
		this_type(other).swap(*this);
		return *this;
	}


	this_type operator&(this_type const &other) const
	{
		return *this;
	}

	void swap(this_type &other)
	{

		std::swap(m_id_min_, other.m_id_min_);
		std::swap(m_id_max_, other.m_id_max_);
		std::swap(m_id_local_min_, other.m_id_local_min_);
		std::swap(m_id_local_max_, other.m_id_local_max_);
		std::swap(m_id_memory_max_, other.m_id_memory_max_);
		std::swap(m_id_memory_min_, other.m_id_memory_min_);
	}

	virtual bool is_valid() const { return true; }

	template<typename TI>
	void dimensions(TI const &d)
	{

		m_id_min_ = m::ID_ZERO;
		m_id_max_ = m_id_min_ + m::pack_index(d);

	}

	index_tuple dimensions() const
	{
		return m::unpack_index(m_id_max_ - m_id_min_);
	}

	template<size_t IFORM>
	size_t max_hash() const
	{
		return m::template max_hash<IFORM>(m_id_memory_min_,
				m_id_memory_max_);
	}

	size_t hash(id_type s) const
	{
		return m::hash(s, m_id_memory_min_, m_id_memory_max_);
	}

	template<size_t IFORM>
	id_type pack_relative_index(index_type i, index_type j,
			index_type k, index_type n = 0) const
	{
		return m::pack_index(nTuple<index_type, 3>({i, j, k}),
				m::template sub_index_to_id<IFORM>(n)) + m_id_min_;
	}


	nTuple<index_type, ndims + 1> unpack_relative_index(id_type s) const
	{
		nTuple<index_type, ndims + 1> res;
		res = m::unpack_index(s - m_id_min_);
		res[ndims] = m::sub_index(s);
		return std::move(res);
	}

	void deploy(size_t const *gw = nullptr)
	{
/**
* Decompose
*/
//
//		if (GLOBAL_COMM.num_of_process() > 1)
//		{
//			auto idx_b = m::unpack_index(m_id_min_);
//
//			auto idx_e = m::unpack_index(m_id_max_);
//
//			GLOBAL_COMM.
//					decompose(ndims, &idx_b[0], &idx_e[0]
//			);
//
//			typename m::index_tuple ghost_width;
//
//			if (gw != nullptr)
//			{
//				ghost_width = gw;
//			}
//			else
//			{
//				ghost_width = DEFAULT_GHOST_WIDTH;
//			}
//
//			for (
//					int i = 0;
//					i < ndims;
//					++i)
//			{
//
//				if (idx_b[i] + 1 == idx_e[i])
//				{
//					ghost_width[i] = 0;
//				}
//				else if (idx_e[i] <= idx_b[i] + ghost_width[i] * 2)
//				{
//					ERROR(
//							"Dimension is to small to split!["
////				" Dimensions= " + type_cast < std::string
////				> (m::unpack_index(
////								m_id_max_ - m_id_min_))
////				+ " , Local dimensions=" + type_cast
////				< std::string
////				> (m::unpack_index(
////								m_id_local_max_ - m_id_local_min_))
////				+ " , Ghost width =" + type_cast
////				< std::string > (ghost_width) +
//									"]");
//				}
//
//			}
//
//			m_id_local_min_ = m::pack_index(idx_b);
//
//			m_id_local_max_ = m::pack_index(idx_e);
//
//			m_id_memory_min_ = m_id_local_min_ - m::pack_index(ghost_width);
//
//			m_id_memory_max_ = m_id_local_max_ + m::pack_index(ghost_width);
//
//
//		}
//		else
		{
			m_id_local_min_ = m_id_min_;

			m_id_local_max_ = m_id_max_;

			m_id_memory_min_ = m_id_local_min_;

			m_id_memory_max_ = m_id_local_max_;

		}

	}

	std::tuple<id_tuple, id_tuple> index_box() const
	{
		return std::make_tuple(m::unpack_index(m_id_min_),
				m::unpack_index(m_id_max_));
	}

	std::tuple<id_tuple, id_tuple> local_index_box() const
	{
		return std::make_tuple(m::unpack_index(m_id_local_min_),
				m::unpack_index(m_id_local_max_));
	}

	template<typename OS>
	OS &print(OS &os) const
	{


//		os << " Dimensions\t= " << dimensions() << "," << std::endl;

		return os;

	}


	auto box() const
	DECL_RET_TYPE(std::forward_as_tuple(m_id_local_min_, m_id_local_max_))

	template<typename T>
	bool in_box(T const &x) const
	{
		auto b = m::unpack_index(m_id_local_min_);
		auto e = m::unpack_index(m_id_local_max_);
		return (b[1] <= x[1]) && (b[2] <= x[2]) && (b[0] <= x[0])  //
				&& (e[1] > x[1]) && (e[2] > x[2]) && (e[0] > x[0]);

	}

	bool in_box(id_type s) const
	{
		return in_box(m::unpack_index(s));
	}


	template<typename ...Args>
	void reset(Args &&...args)
	{
		this_type(m::pack(args)...).swap(*this);
	}


	range_type range(int nid = 0) const
	{
		return range_type(m_id_local_min_, m_id_local_max_, nid);
	}

	template<int I> range_type make_range() const
	{
		return range_type(m_id_local_min_, m_id_local_max_, m::sub_index_to_id<I>());
	}


//	range_type range(point_type const &min, point_type const &max,
//			int nid = 0) const
//	{
////		geometry::model::Box<point_type> b;
////		bool success = geometry::intersection(
////				geometry::make_box(point(m_id_local_min_),
////						point(m_id_local_min_)), geometry::make_box(min, max),
////				b);
////		if (success)
////		{
////			return range_type(
////					traits::get<0>(
////							coordinates_global_to_local(traits::get<0>(b), nid)),
////					traits::get<1>(
////							coordinates_global_to_local(traits::get<1>(b), nid)),
////					nid);
////		}
////		else
////		{
////		}
//
//	}
//
};//struct StructuredMesh


namespace traits
{

template<typename TAG>
struct point_type<Topology<TAG> >
{
	typedef nTuple<Real, 3> type;
};
} //namespace traits
} // namespace topology

typedef Topology<topology::tags::CoRectMesh> CoRectMesh;
typedef Topology<topology::tags::Curvilinear> Curvilinear;
typedef Topology<topology::tags::RectMesh> RectMesh;

template<>
struct Topology<topology::tags::CoRectMesh> : public topology::StructuredMesh { };

template<>
struct Topology<topology::tags::RectMesh> : public topology::StructuredMesh { };

template<>
struct Topology<topology::tags::Curvilinear> : public topology::StructuredMesh { };

} // namespace simpla

#endif //SIMPLA_TOPOLOGY_H
