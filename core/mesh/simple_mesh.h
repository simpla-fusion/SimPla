/**
 * @file  simple_mesh.h
 *
 *  Created on: 2014年11月5日
 *      Author: salmon
 */

#ifndef CORE_DIFF_GEOMETRY_SIMPLE_MESH_H_
#define CORE_DIFF_GEOMETRY_SIMPLE_MESH_H_

#include "structured/topology/structured.h"

namespace simpla
{

typedef RectMesh SimpleMesh;

//template<typename ...>class _Field;
//template<typename ...>class Expression;
//
//struct SimpleMesh: public enable_create_from_this<SimpleMesh>
//{
//public:
//
//	typedef SimpleMesh this_type;
//
//	static constexpr size_t ndims = 3;
//
//	typedef size_t id_type;
//
//	typedef sp_ndarray_range<ndims, size_t> range_type;
//
//	typedef nTuple<size_t, ndims> index_tuple;
//
//	typedef nTuple<Real, ndims> coordinate_tuple;
//
//	typedef this_type geometry_type;
//
//	typedef this_type topology_type;
//
//	template<typename TV> using field_value_type = TV;
//
//private:
//
//	coordinate_tuple m_xmin_, m_xmax_, m_dx_;
//	Real m_dt_ = 1.0;
//	index_tuple m_global_dimensions_;
//	index_tuple m_global_offset_;
//	index_tuple m_local_dimensions_;
//	index_tuple m_local_offset_;
//	index_tuple m_count_;
//	index_tuple m_local_strides_;
//	index_tuple m_ghost_width_;
//	index_tuple m_grain_size_;
//
//	DataSpace m_dataspace_;
//
//	std::vector<mpi_ghosts_shape_s> m_ghosts_shape_;
//
//public:
//	SimpleMesh()
//	{
//	}
//	template<typename TI, typename TX>
//	SimpleMesh(TI const & dimensions, TX const& xmin, TX const& xmax) :
//			m_xmin_(xmin), m_xmax_(xmax), m_count_(dimensions)
//	{
//		m_global_offset_ = 0;
//		m_ghost_width_ = 0;
//		m_grain_size_ = (m_ghost_width_ + 1) * 2;
//
//		m_dx_ = (m_xmax_ - m_xmin_) / m_count_;
//
//	}
//	SimpleMesh(SimpleMesh const & other) :
//			m_xmin_(other.m_xmin_), m_xmax_(other.m_xmax_), m_dx_(other.m_dx_),
//
//			m_global_dimensions_(other.m_global_dimensions_),
//
//			m_global_offset_(other.m_global_offset_),
//
//			m_local_dimensions_(other.m_local_dimensions_),
//
//			m_local_offset_(other.m_local_offset_),
//
//			m_count_(other.m_count_),
//
//			m_ghost_width_(other.m_ghost_width_),
//
//			m_grain_size_(other.m_grain_size_),
//
//			m_dataspace_(other.m_dataspace_)
//	{
//	}
//
//	SimpleMesh(SimpleMesh & other, op_split);
//
//	~SimpleMesh()
//	{
//	}
//	void swap(SimpleMesh & other)
//	{
//		std::swap(m_xmin_, other.m_xmin_);
//		std::swap(m_xmax_, other.m_xmax_);
//		std::swap(m_dx_, other.m_dx_);
//
//		std::swap(m_global_dimensions_, other.m_global_dimensions_);
//		std::swap(m_global_offset_, other.m_global_offset_);
//		std::swap(m_local_dimensions_, other.m_local_dimensions_);
//		std::swap(m_local_offset_, other.m_local_offset_);
//		std::swap(m_count_, other.m_count_);
//		std::swap(m_ghost_width_, other.m_ghost_width_);
//		std::swap(m_grain_size_, other.m_grain_size_);
//		std::swap(m_dataspace_, other.m_dataspace_);
//
//	}
//	SimpleMesh & operator=(SimpleMesh const & other)
//	{
//		SimpleMesh(other).swap(*this);
//		return *this;
//	}
//	std::string get_type_as_string() const
//	{
//		return "SimpleMesh";
//	}
//
//	bool is_divisible() const
//	{
//		return false;
//	}
//
//	template<typename TDict>
//	void load(TDict const & dict)
//	{
//	}
//
//	template<typename OS>
//	OS & print(OS & os) const
//	{
//		os << "{"
//
//		<< "  xmin=" << m_xmin_
//
//		<< ", xmax=" << m_xmax_
//
//		<< ", dimensions=" << m_count_
//
//		<< " }";
//		return os;
//
//	}
//
//	Real dt() const
//	{
//		return m_dt_;
//	}
//	void dt(Real pdt)
//	{
//		m_dt_ = pdt;
//	}
//	template<typename T1>
//	void dimensions(T1 const & d)
//	{
//		m_global_dimensions_ = d;
//	}
//
//	index_tuple const & dimensions() const
//	{
//		return m_global_dimensions_;
//	}
//	template<typename T1>
//	void ghost_width(T1 const & d)
//	{
//		m_ghost_width_ = d;
//	}
//
//	index_tuple const & ghost_width() const
//	{
//		return m_ghost_width_;
//	}
//	template<typename T1, typename T2>
//	void extents(T1 const & xmin, T2 const & xmax)
//	{
//		m_xmin_ = xmin;
//		m_xmax_ = xmax;
//	}
//
//	std::pair<coordinate_tuple, coordinate_tuple> extents() const
//	{
//		return std::make_pair(m_xmin_, m_xmax_);
//	}
//
//	void deploy()
//	{
//
//		DataSpace(ndims, &m_global_dimensions_[0]).swap(m_dataspace_);
//		m_count_ = m_global_dimensions_;
//		m_global_offset_ = 0;
//		if (GLOBAL_COMM.num_of_process()>1)
//		{
//			GLOBAL_COMM.decompose(ndims, &m_global_offset_[0], &m_count_[0]);
//		}
//
//		CHECK(m_count_);
//
//		m_local_offset_ = m_ghost_width_;
//
//		m_local_dimensions_ = m_count_ + m_ghost_width_ * 2;
//
//		m_local_strides_[ndims - 1] = 1;
//
//		if (ndims > 1)
//		{
//			for (int i = ndims - 2; i >= 0; --i)
//			{
//				m_local_strides_[i] = m_local_dimensions_[i + 1]
//						* m_local_strides_[i + 1];
//			}
//		}
//		m_dataspace_
//
//		.select_hyperslab(&m_global_offset_[0], nullptr, &m_count_[0], nullptr)
//
//		.convert_to_local(&m_ghost_width_[0]);
//
//		get_ghost_shape(ndims, &m_local_dimensions_[0], &m_local_offset_[0],
//				nullptr, &m_count_[0], nullptr, &m_ghost_width_[0],
//				&m_ghosts_shape_);
//
//	}
//	std::vector<mpi_ghosts_shape_s> const & ghost_shape() const
//	{
//		return m_ghosts_shape_;
//	}
//
//	size_t max_hash() const
//	{
//		CHECK(m_local_dimensions_);
//		size_t res = 1;
//		for (int i = 0; i < ndims; ++i)
//		{
//			res *= m_local_dimensions_[i];
//		}
//		return res;
//	}
//
//	size_t hash(index_tuple const & s) const
//	{
//		return inner_product(s + m_local_offset_ - m_global_offset_,
//				m_local_strides_);
//	}
//
//	size_t hash(size_t N0, size_t N1) const
//	{
//		return (N0 + m_local_offset_[0] - m_global_offset_[0])
//				* m_local_strides_[0]
//				+ (N1 + m_local_offset_[1] - m_global_offset_[1])
//						* m_local_strides_[1];
//	}
//
//	size_t hash(size_t N0, size_t N1, size_t N2) const
//	{
//		return (N0 + m_local_offset_[0] - m_global_offset_[0])
//				* m_local_strides_[0]
//				+ (N1 + m_local_offset_[1] - m_global_offset_[1])
//						* m_local_strides_[1]
//				+ (N2 + m_local_offset_[2] - m_global_offset_[2])
//						* m_local_strides_[2];
//	}
//	DataSpace const& dataspace() const
//	{
//		return m_dataspace_;
//	}
//	index_tuple coordinates_to_index(coordinate_tuple const &x) const
//	{
//		index_tuple res;
//		res = (x - m_xmin_) / m_dx_;
//		return std::move(res);
//	}
//	coordinate_tuple index_to_coordinates(index_tuple const &i) const
//	{
//		coordinate_tuple res;
//		res = i * m_dx_ + m_xmin_;
//		return std::move(res);
//	}
//	index_tuple coordinates_to_id(coordinate_tuple const &x) const
//	{
//		index_tuple res;
//		res = (x - m_xmin_) / m_dx_;
//		return std::move(res);
//	}
//	coordinate_tuple id_to_coordinates(index_tuple const &i) const
//	{
//		coordinate_tuple res;
//		res = i * m_dx_ + m_xmin_;
//		return std::move(res);
//	}
//
//	this_type & self()
//	{
//		return *this;
//	}
//	this_type const& self() const
//	{
//		return *this;
//	}
//
//	range_type range() const
//	{
//		return range_type(m_global_offset_, m_global_offset_ + m_count_);
//	}
//	template<typename ...Args>
//	size_t hash(Args &&...args) const
//	{
//		return 1;
//	}
//
//	template<typename T>
//	range_type select(T imin, T imax) const
//	{
//		return range_type(imin, imax);
//
//	}
//
//private:
//	template<typename TOP, typename ... Args>
//	constexpr auto calculate_(TOP op, Args &&...args,
//			index_tuple const &s) const
//			DECL_RET_TYPE (op((std::forward<Args>(args), s)...))
//
////	template<typename TOP, typename TL, typename TR>
////	inline auto calculate_(TOP op, TL & l, TR &r, id_type const &s) const
////	DECL_RET_TYPE( op(( (l),s),(r,s) ) )
//
//public:
//
//	template<typename TOP, typename TL>
//	constexpr auto calculate(_Field<Expression<TOP, TL> > const & f,
//			index_tuple const &s) const
//			DECL_RET_TYPE((calculate_(f.op_,f.lhs,s)))
//
//	template<typename TOP, typename TL, typename TR>
//	constexpr auto calculate(_Field<Expression<TOP, TL, TR> > const & f,
//			index_tuple const &s) const
//			DECL_RET_TYPE((calculate_(f.op_,f.lhs,f.rhs,s)))
//
//	template<typename TC, typename TD>
//	constexpr auto calculate(_Field<TC, TD> const & f,
//			index_tuple const &s) const
//			DECL_RET_TYPE ((f[s]))
//
//	template<typename T>
//	constexpr T const& calculate(T const & v, index_tuple const &s) const
//	{
//		return v;
//	}
//
//	template<typename TOP, typename TL, typename TR>
//	void calculate(
//			_Field<AssignmentExpression<TOP, TL, TR> > const & fexpr) const
//	{
////		foreach(fexpr.op_, fexpr.lhs, fexpr.rhs);
//	}
//
//	template<typename T>
//	auto calculate(T const & v, index_tuple const &s) const
//	DECL_RET_TYPE (((v, s)))
//
//	coordinate_tuple coordinates(index_tuple const & s) const
//	{
//		coordinate_tuple res;
////		res = (s - m_imin_) * m_dx_ + m_xmin_;
//		return res;
//	}
//	template<typename TV>
//	constexpr TV sample(index_tuple const &s, TV const &v) const
//	{
//		return v;
//	}
//
//	template<typename TD>
//	auto gather(TD const & d,
//			coordinate_tuple const & x) const->decltype(d[std::declval<index_tuple>()])
//	{
//		index_tuple r;
//		r = ((x - m_xmin_) / m_dx_ + 0.5);
//
//		return d[r];
//	}
//
//	template<typename TD, typename TV>
//	void scatter(TD & d, coordinate_tuple const &x, TV const & v) const
//	{
//		index_tuple r;
//		r = ((x - m_xmin_) / m_dx_ + 0.5);
//
//		d[r] += v;
//	}
//
//};
//
//std::ostream & operator<<(std::ostream & os, SimpleMesh const & mesh)
//{
//	return mesh.print(os);
//}
}
// namespace simpla

#endif /* CORE_DIFF_GEOMETRY_SIMPLE_MESH_H_ */
