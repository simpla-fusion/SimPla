/**
 * @file  simple_mesh.h
 *
 *  Created on: 2014年11月5日
 *      Author: salmon
 */

#ifndef CORE_DIFF_GEOMETRY_SIMPLE_MESH_H_
#define CORE_DIFF_GEOMETRY_SIMPLE_MESH_H_

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <iterator>
#include "../dataset/dataspace.h"
#include "../gtl/enable_create_from_this.h"
#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"
#include "../gtl/iterator/range.h"
#include "../gtl/iterator/sp_ndarray_iterator.h"
#include "../utilities/utilities.h"

#include "../parallel/mpi_comm.h"
namespace simpla
{
template<typename ...>class _Field;
template<typename ...>class Expression;

struct SimpleMesh: public enable_create_from_this<SimpleMesh>
{
public:

	typedef SimpleMesh this_type;

	static constexpr size_t m_ndims_ = 3;

	typedef size_t id_type;

	typedef sp_ndarray_range<m_ndims_, size_t> range_type;

	typedef nTuple<size_t, m_ndims_> index_tuple;

	typedef nTuple<Real, m_ndims_> coordinates_type;

	typedef this_type geometry_type;

	typedef this_type topology_type;

	template<typename TV> using field_value_type = TV;

private:

	coordinates_type m_xmin_, m_xmax_, m_dx_;
	index_tuple m_dimensions_;
	index_tuple m_offset_;
	index_tuple m_count_;
	index_tuple m_strides_;
	index_tuple m_ghost_width_;

	index_tuple m_grain_size_;

	DataSpace m_dataspace_;

public:
	SimpleMesh()
	{
	}
	template<typename TI, typename TX>
	SimpleMesh(TI const & dimensions, TX const& xmin, TX const& xmax)
			: m_xmin_(xmin), m_xmax_(xmax), m_count_(dimensions)
	{
		m_offset_ = 0;
		m_ghost_width_ = 0;
		m_grain_size_ = (m_ghost_width_ + 1) * 2;

		m_dx_ = (m_xmax_ - m_xmin_) / m_count_;

	}
	SimpleMesh(SimpleMesh const & other)
			: m_xmin_(other.m_xmin_), m_xmax_(other.m_xmax_), m_dx_(
					other.m_dx_), m_dimensions_(other.m_dimensions_), m_offset_(
					other.m_offset_), m_count_(other.m_count_), m_grain_size_(
					other.m_grain_size_), m_dataspace_(other.m_dataspace_)
	{
	}

	SimpleMesh(SimpleMesh & other, op_split);

	~SimpleMesh()
	{
	}
	void swap(SimpleMesh & other)
	{
		std::swap(m_xmin_, other.m_xmin_);
		std::swap(m_xmax_, other.m_xmax_);
		std::swap(m_dx_, other.m_dx_);
		std::swap(m_dimensions_, other.m_dimensions_);
	}
	SimpleMesh & operator=(SimpleMesh const & other)
	{
		SimpleMesh(other).swap(*this);
		return *this;
	}
	std::string get_type_as_string() const
	{
		return "SimpleMesh";
	}

	bool is_divisible() const
	{
		return false;
	}

	template<typename TDict>
	void load(TDict const & dict)
	{
	}

	template<typename OS>
	OS & print(OS & os) const
	{
		os << "{"

		<< "  xmin=" << m_xmin_

		<< ", xmax=" << m_xmax_

		<< ", dimensions=" << m_count_

		<< " }";
		return os;

	}
	template<typename T1>
	void dimensions(T1 const & d)
	{
		m_count_ = d;
	}

	index_tuple const & dimensions() const
	{
		return m_count_;
	}
	template<typename T1>
	void ghost_width(T1 const & d)
	{
		m_ghost_width_ = d;
	}

	index_tuple const & ghost_width() const
	{
		return m_ghost_width_;
	}
	template<typename T1, typename T2>
	void extents(T1 const & xmin, T2 const & xmax)
	{
		m_xmin_ = xmin;
		m_xmax_ = xmax;
	}

	std::pair<coordinates_type, coordinates_type> extents() const
	{
		return std::make_pair(m_xmin_, m_xmax_);
	}

	void deploy()
	{

		DataSpace ds;

		if (m_dataspace_.is_valid())
		{
			m_dataspace_.swap(ds);
		}
		else
		{
			DataSpace(m_ndims_, &m_count_[0]).swap(ds);
		}

		if (GLOBAL_COMM.num_of_process()>1)
		{
			GLOBAL_COMM.decompose(m_ndims_, &m_offset_[0], &m_count_[0]);

			ds.select_hyperslab(&m_offset_[0], nullptr, &m_count_[0], nullptr);

			ds.create_distributed_space(&m_ghost_width_[0]).swap(m_dataspace_);
		}
		else
		{
			ds.swap(m_dataspace_);
		}

		std::tie(std::ignore, m_dimensions_, m_offset_, std::ignore, m_count_,
				std::ignore) = m_dataspace_.shape();

		m_strides_[m_ndims_ - 1] = 1;

		if (m_ndims_ > 1)
		{
			for (int i = m_ndims_ - 2; i >= 0; --i)
			{
				m_strides_[i] = m_dimensions_[i + 1] * m_strides_[i + 1];
			}
		}

	}

	size_t max_hash() const
	{
		return NProduct(m_count_);
	}

	size_t hash(index_tuple const & s) const
	{
		return inner_product(s - m_offset_, m_strides_);
	}

	size_t hash(size_t N0, size_t N1) const
	{
		return (N0 - m_offset_[0]) * m_strides_[0]
				+ (N1 - m_offset_[1]) * m_strides_[1];
	}

	size_t hash(size_t N0, size_t N1, size_t N2) const
	{
		return (N0 - m_offset_[0]) * m_strides_[0]
				+ (N1 - m_offset_[1]) * m_strides_[1]
				+ (N2 - m_offset_[2]) * m_strides_[2];
	}
	DataSpace const& dataspace() const
	{
		return m_dataspace_;
	}

	index_tuple coordinates_to_id(coordinates_type const &x) const
	{
		index_tuple res;
		res = (x - m_xmin_) / m_dx_;
		return std::move(res);
	}
	coordinates_type id_to_coordinates(index_tuple const &i) const
	{
		coordinates_type res;
		res = i * m_dx_ + m_xmin_;
		return std::move(res);
	}

	this_type & self()
	{
		return *this;
	}
	this_type const& self() const
	{
		return *this;
	}

	range_type range() const
	{
		return range_type(m_offset_, m_count_);
	}
	template<typename ...Args>
	size_t hash(Args &&...args) const
	{
		return 1;
	}

private:
	template<typename TOP, typename ... Args>
	constexpr auto calculate_(TOP op, Args &&...args,
			index_tuple const &s) const
			DECL_RET_TYPE (op(get_value(std::forward<Args>(args), s)...))

//	template<typename TOP, typename TL, typename TR>
//	inline auto calculate_(TOP op, TL & l, TR &r, id_type const &s) const
//	DECL_RET_TYPE( op(get_value( (l),s),get_value(r,s) ) )

public:

	template<typename TOP, typename TL>
	constexpr auto calculate(_Field<Expression<TOP, TL> > const & f,
			index_tuple const &s) const
			DECL_RET_TYPE((calculate_(f.op_,f.lhs,s)))

	template<typename TOP, typename TL, typename TR>
	constexpr auto calculate(_Field<Expression<TOP, TL, TR> > const & f,
			index_tuple const &s) const
			DECL_RET_TYPE((calculate_(f.op_,f.lhs,f.rhs,s)))

	template<typename TC, typename TD>
	constexpr auto calculate(_Field<TC, TD> const & f,
			index_tuple const &s) const
			DECL_RET_TYPE ((f[s]))

	template<typename T>
	constexpr T const& calculate(T const & v, index_tuple const &s) const
	{
		return v;
	}

	template<typename TOP, typename TL, typename TR>
	void calculate(
			_Field<AssignmentExpression<TOP, TL, TR> > const & fexpr) const
	{
//		foreach(fexpr.op_, fexpr.lhs, fexpr.rhs);
	}

	template<typename T>
	auto calculate(T const & v, index_tuple const &s) const
	DECL_RET_TYPE ((get_value(v, s)))

	coordinates_type coordinates(index_tuple const & s) const
	{
		coordinates_type res;
//		res = (s - m_imin_) * m_dx_ + m_xmin_;
		return res;
	}
	template<typename TV>
	constexpr TV sample(index_tuple const &s, TV const &v) const
	{
		return v;
	}

	template<typename TD>
	auto gather(TD const & d,
			coordinates_type const & x) const->decltype(d[std::declval<index_tuple>()])
	{
		index_tuple r;
		r = ((x - m_xmin_) / m_dx_ + 0.5);

		return d[r];
	}

	template<typename TD, typename TV>
	void scatter(TD & d, coordinates_type const &x, TV const & v) const
	{
		index_tuple r;
		r = ((x - m_xmin_) / m_dx_ + 0.5);

		d[r] += v;
	}

};

std::ostream & operator<<(std::ostream & os, SimpleMesh const & mesh)
{
	return mesh.print(os);
}
}
// namespace simpla

#endif /* CORE_DIFF_GEOMETRY_SIMPLE_MESH_H_ */
