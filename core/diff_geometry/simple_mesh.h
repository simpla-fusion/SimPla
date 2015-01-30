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
#include "../gtl/enable_create_from_this.h"
#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"
#include "../utilities/utilities.h"
namespace simpla
{
template<typename ...>class _Field;
template<typename ...>class Expression;

struct SimpleMesh: public enable_create_from_this<SimpleMesh>
{
public:

	typedef SimpleMesh this_type;

	static constexpr size_t ndims = 3;

	typedef nTuple<size_t, ndims> id_type;

	typedef nTuple<Real, ndims> coordinates_type;

	typedef this_type geometry_type;

	typedef this_type topology_type;

	template<typename TV> using field_value_type=TV;

private:

	coordinates_type xmin_, xmax_, dx_;

	nTuple<size_t, ndims> offset_, count_, strides_;

public:
	SimpleMesh()
	{
		xmin_ = 0;
		xmax_ = 1;
		offset_ = 0;
		count_ = 10;
		update();
	}
	SimpleMesh(coordinates_type const & xmin, coordinates_type const & xmax,
			nTuple<size_t, ndims> const& count,
			nTuple<size_t, ndims> const&offset) :
			xmin_(xmin), xmax_(xmax), count_(count), offset_(offset)
	{
		update();
	}
	SimpleMesh(SimpleMesh const & other) :
			xmin_(other.xmin_), xmax_(other.xmax_), offset_(other.offset_), count_(
					other.count_)
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

	virtual std::basic_ostream<char>& print(std::basic_ostream<char>& os) const
	{
		os << "{ xmin=" << xmin_ << ", xmax=" << xmax_ << ", count=" << count_
				<< ", offset=" << offset_ << " }";
		return os;

	}

	template<typename T1, typename T2>
	void extents(T1 const & xmin, T2 const & xmax)
	{
		xmin_ = xmin;
		xmax_ = xmax;
	}

	std::pair<coordinates_type, coordinates_type> extents() const
	{
		return std::make_pair(xmin_, xmax_);
	}

	template<typename T1>
	void dimensions(T1 const & d)
	{
		count_ = d;
	}

	nTuple<size_t, ndims> const &dimensions() const
	{
		return count_;
	}

	template<typename T1>
	void offset(T1 const & d)
	{
		offset_ = d;
	}

	nTuple<size_t, ndims> const &offset() const
	{
		return offset_;
	}

	static constexpr size_t array_order = C_ORDER;

	void update()
	{
		dx_ = (xmax_ - xmin_) / count_;

		strides_[ndims - 1] = 1;
		if (ndims > 1)
		{
			for (int i = ndims - 2; i >= 0; --i)
			{
				strides_[i] = count_[i + 1] * strides_[i + 1];
			}
		}
	}

	/**@name range concept
	 * @{
	 */
	SimpleMesh(SimpleMesh &, op_split)
	{
		UNIMPLEMENTED;
	}

	bool empty() const
	{
		return false;
	}

	bool is_divisible() const
	{
		return false;
	}
	/**@}*/
	this_type & self()
	{
		return *this;
	}
	this_type const& self() const
	{
		return *this;
	}

	struct const_iterator
	{
		id_type begin_, end_, self_;

		const_iterator(id_type const &b, id_type const &e) :
				begin_(b), end_(e), self_(b)
		{
		}
		const_iterator(id_type const &b, id_type const &e, id_type const &s) :
				begin_(b), end_(e), self_(s)
		{
		}
		const_iterator(const_iterator const& other) :
				begin_(other.begin_), end_(other.end_), self_(other.self_)
		{

		}
		const_iterator(const_iterator && other) :
				begin_(other.begin_), end_(other.end_), self_(other.self_)
		{

		}
		~const_iterator()
		{
		}

		id_type const & operator *() const
		{
			return self_;
		}
		const_iterator & operator++()
		{
			int n = ndims - 1;

			++self_[n];

			while (n > 0 && self_[n] >= end_[n])
			{
				self_[n] = begin_[n];
				++self_[n - 1];
				--n;
			}

//			if (self_[0] >= end_[0])
//			{
//				self_[0] = begin_[0];
//			}
			return *this;
		}
		const_iterator operator++(int) const
		{
			const_iterator res(*this);
			++res;
			return std::move(res);
		}

		bool operator==(const_iterator const & other) const
		{
//			bool res = false;
//
//			for (int i = 0; i < ndims; ++i)
//			{
//				if (self_[i] != other.self_[i])
//				{
//					res = true;
//					break;
//				}
//			}

			return self_ == other.self_;
		}

		bool operator!=(const_iterator const & other) const
		{
//			bool res = false;
//
//			for (int i = 0; i < ndims; ++i)
//			{
//				if (self_[i] != other.self_[i])
//				{
//					res = true;
//					break;
//				}
//			}

			return !operator==(other);
		}
	};

	const_iterator begin() const
	{
		id_type e;
		e = offset_ + count_;
		return std::move(const_iterator(offset_, e));
	}

	const_iterator end() const
	{
		id_type e, s;
		e = offset_ + count_;
		s = e - 1;
		const_iterator res(offset_, e, s);
		++res;
		return std::move(res);
	}

//	template<typename ...Args>
//	size_t hash(Args && ... args) const
//	{
//		return root().hash(std::forward<Args>(args)...);
//	}

	size_t hash(id_type const & s) const
	{

		return inner_product(s - offset_, strides_);
	}

	size_t max_hash() const
	{
		size_t res = strides_[0] * count_[0];
		if (!is_root())
		{
			res = root().max_hash();
		}
		return res;
	}

	coordinates_type coordinates(id_type const & s) const
	{
		coordinates_type res;
		res = (s - offset_) * dx_ + xmin_;
		return res;
	}
	template<typename TV>
	TV sample(id_type const &s, TV const &v) const
	{
		return v;
	}

	template<typename TD>
	auto gather(TD const & d,
			coordinates_type const & x) const->decltype(d[std::declval<id_type>()])
	{
		id_type r;
		r = ((x - xmin_) / dx_ + 0.5);

		return d[r];
	}

	template<typename TD, typename TV>
	void scatter(TD & d, coordinates_type const &x, TV const & v) const
	{
		id_type r;
		r = ((x - xmin_) / dx_ + 0.5);

		d[r] += v;
	}

	template<typename TFun, typename ...Args>
	void foreach(TFun const &fun, Args && ...args) const
	{
		for (auto const &s : *this)
		{
			fun(get_value(std::forward<Args>(args),s)...);
		}
	}

private:
	template<typename TOP, typename ... Args>
	inline auto calculate_(TOP op, Args &&...args, id_type const &s) const
	DECL_RET_TYPE (op(get_value(std::forward<Args>(args), s)...))

//	template<typename TOP, typename TL, typename TR>
//	inline auto calculate_(TOP op, TL & l, TR &r, id_type const &s) const
//	DECL_RET_TYPE( op(get_value( (l),s),get_value(r,s) ) )

public:

	template<typename TOP, typename TL>
	auto calculate(_Field<Expression<TOP, TL> > const & f,
			id_type const &s) const
			DECL_RET_TYPE((calculate_(f.op_,f.lhs,s)))

	template<typename TOP, typename TL, typename TR>
	auto calculate(_Field<Expression<TOP, TL, TR> > const & f,
			id_type const &s) const
			DECL_RET_TYPE((calculate_(f.op_,f.lhs,f.rhs,s)))

	template<typename TC, typename TD>
	auto calculate(_Field<TC, TD> const & f, id_type const &s) const
	DECL_RET_TYPE ((f[s]))

	template<typename T, size_t ...N>
	nTuple<T, N...> const& calculate(nTuple<T, N...> const & v,
			id_type const &s) const
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
	auto calculate(T const & v, id_type const &s) const
	DECL_RET_TYPE ((get_value(v, s)))

};

std::ostream & operator<<(std::ostream & os, SimpleMesh const & mesh)
{
	return mesh.print(os);
}
}
// namespace simpla

#endif /* CORE_DIFF_GEOMETRY_SIMPLE_MESH_H_ */
