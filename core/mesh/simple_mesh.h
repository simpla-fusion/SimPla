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
#include "../gtl/enable_create_from_this.h"
#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"
#include "../gtl/iterator/range.h"
#include "../gtl/iterator/sp_ndarray_iterator.h"
#include "../utilities/utilities.h"
namespace simpla
{
template<typename ...>class _Field;
template<typename ...>class Expression;

struct SimpleMesh:	public sp_ndarray_range<3, size_t>,
					public enable_create_from_this<SimpleMesh>
{
public:

	typedef SimpleMesh this_type;

	static constexpr size_t ndims = 3;

	typedef size_t id_type;

	typedef sp_ndarray_range<ndims, size_t> range_type;

	typedef nTuple<size_t, ndims> indices_type;

	typedef nTuple<Real, ndims> coordinates_type;

	typedef this_type geometry_type;

	typedef this_type topology_type;

	template<typename TV> using field_value_type = TV;

private:

	coordinates_type m_xmin_, m_xmax_, dx_;

	nTuple<size_t, ndims> m_imin_, m_imax_;

public:
	SimpleMesh()
	{
		update();
	}
	SimpleMesh(coordinates_type const & xmin, coordinates_type const & xmax,
			nTuple<size_t, ndims> const& imin,
			nTuple<size_t, ndims> const& imax)
			: range_type(imin, imax), m_xmin_(xmin), m_xmax_(xmax)
	{
		update();
	}
	SimpleMesh(SimpleMesh const & other)
			: range_type(other), m_xmin_(other.m_xmin_), m_xmax_(other.m_xmax_)
	{
		update();
	}

	~SimpleMesh()
	{
	}

	std::string get_type_as_string() const
	{
		return "SimpleMesh";
	}

	template<typename OS>
	OS & print(OS & os) const
	{
		os << "{"

		<< "  xmin=" << m_xmin_

		<< ", xmax=" << m_xmax_

		<< ", imin=" << *range_type::begin()

		<< ", imax=" << *range_type::end()

		<< " }";
		return os;

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

	void update()
	{
		dx_ = (m_xmax_ - m_xmin_) / (*range_type::end() - *range_type::begin());
	}
	indices_type coordinates_to_id(coordinates_type const &x) const
	{
		indices_type res;
		res = (x - m_xmin_) / dx_;
		return std::move(res);
	}
	coordinates_type id_to_coordinates(indices_type const &i) const
	{
		coordinates_type res;
		res = i * dx_ + m_xmin_;
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
		return *dynamic_cast<range_type const *>(this);
	}

private:
	template<typename TOP, typename ... Args>
	constexpr auto calculate_(TOP op, Args &&...args,
			indices_type const &s) const
			DECL_RET_TYPE (op(get_value(std::forward<Args>(args), s)...))

//	template<typename TOP, typename TL, typename TR>
//	inline auto calculate_(TOP op, TL & l, TR &r, id_type const &s) const
//	DECL_RET_TYPE( op(get_value( (l),s),get_value(r,s) ) )

public:

	template<typename TOP, typename TL>
	constexpr auto calculate(_Field<Expression<TOP, TL> > const & f,
			indices_type const &s) const
			DECL_RET_TYPE((calculate_(f.op_,f.lhs,s)))

	template<typename TOP, typename TL, typename TR>
	constexpr auto calculate(_Field<Expression<TOP, TL, TR> > const & f,
			indices_type const &s) const
			DECL_RET_TYPE((calculate_(f.op_,f.lhs,f.rhs,s)))

	template<typename TC, typename TD>
	constexpr auto calculate(_Field<TC, TD> const & f,
			indices_type const &s) const
			DECL_RET_TYPE ((f[s]))

	template<typename T>
	constexpr T const& calculate(T const & v, indices_type const &s) const
	{
		return v;
	}

	template<typename TOP, typename TL, typename TR>
	void calculate(
			_Field<AssignmentExpression<TOP, TL, TR> > const & fexpr) const
	{
		foreach(fexpr.op_, fexpr.lhs, fexpr.rhs);
	}

	template<typename T>
	auto calculate(T const & v, indices_type const &s) const
	DECL_RET_TYPE ((get_value(v, s)))

	coordinates_type coordinates(indices_type const & s) const
	{
		coordinates_type res;
		res = (s - m_imin_) * dx_ + m_xmin_;
		return res;
	}
	template<typename TV>
	constexpr TV sample(indices_type const &s, TV const &v) const
	{
		return v;
	}

	template<typename TD>
	auto gather(TD const & d,
			coordinates_type const & x) const->decltype(d[std::declval<indices_type>()])
	{
		indices_type r;
		r = ((x - m_xmin_) / dx_ + 0.5);

		return d[r];
	}

	template<typename TD, typename TV>
	void scatter(TD & d, coordinates_type const &x, TV const & v) const
	{
		indices_type r;
		r = ((x - m_xmin_) / dx_ + 0.5);

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
