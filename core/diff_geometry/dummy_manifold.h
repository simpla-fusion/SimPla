/**
 * @file  dummy_manifold.h
 *
 *  Created on: 2014年11月5日
 *      Author: salmon
 */

#ifndef CORE_DIFF_GEOMETRY_DUMMY_MANIFOLD_H_
#define CORE_DIFF_GEOMETRY_DUMMY_MANIFOLD_H_

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <type_traits>

#include "../diff_geometry/domain.h"
#include "../parallel/block_range.h"
#include "../utilities/primitives.h"
#include "../utilities/type_traits.h"

namespace simpla
{
template<typename ...>class _Field;
template<typename ...>class Expression;

struct DummyManifold: public std::enable_shared_from_this<DummyManifold>
{
public:

	typedef DummyManifold this_type;

	typedef size_t index_type;

	typedef Real coordinates_type;

	typedef BlockRange<size_t> range_type;

	typedef this_type geometry_type;

	typedef this_type topology_type;

	static constexpr size_t ndims = 1;

private:

	coordinates_type b_, e_, dx_;
	range_type range_;
public:

	template<typename ...Args>
	DummyManifold(Args && ... args) :
			range_(std::forward<Args>(args)...)
	{

	}
	~DummyManifold()
	{
	}

	void swap(this_type & that)
	{
		std::swap(b_, that.b_);
		std::swap(e_, that.e_);
		std::swap(dx_, that.dx_);
		range_.swap(that.range_);
	}

	template<typename ...Args>
	void dimensions(Args && ... args)
	{
		range_type(std::forward<Args>(args)...).swap(range_);
	}

	template<typename ...Args>
	auto hash(Args && ... args) const
	DECL_RET_TYPE((range_.hash(std::forward<Args>(args)...)))

	auto max_hash() const
	DECL_RET_TYPE((range_.max_hash( )))

	void extents(coordinates_type b, coordinates_type e)
	{
		b_ = b;
		e_ = e;
		dx_ = (e_ - b_) / range_.max_hash();
	}

	std::pair<coordinates_type, coordinates_type> extents() const
	{
		return std::make_pair(b_, e_);
	}

	void update()
	{
	}

	coordinates_type coordinates(index_type const & s) const
	{
		coordinates_type res;
		res = s * dx_;
		return res;
	}
	template<typename TV, size_t IFORM>
	TV sample(std::integral_constant<size_t, IFORM>, index_type s,
			TV const &v) const
	{
		return v;
	}
	// interpolator
	template<typename TD>
	auto gather(TD const & d,
			coordinates_type x) const->decltype(d[std::declval<index_type>()])
	{
		index_type r = static_cast<index_type>((x - b_) / dx_ + 0.5);

		return d[r];
	}

	template<typename TD, typename TV>
	void scatter(TD & d, coordinates_type x, TV const & v) const
	{
		index_type r = static_cast<index_type>((x - b_) / dx_ + 0.5);

		d[r] += v;
	}

	// topology
	template<typename ... Args>
	int dataset_shape(Args &&...args) const
	{
		return 0;
	}

	template<size_t IFORM, typename ...Args>
	range_type select(Args &&... args) const
	{
		return range_;
	}

// diff_scheme
private:
	template<typename TOP, typename TL>
	inline auto calculate_(TOP op, TL const& f, index_type s) const
	DECL_RET_TYPE( op(get(f,s) ) )

	template<typename TOP, typename TL, typename TR>
	inline auto calculate_(TOP op, TL const& l, TR const &r, index_type s) const
	DECL_RET_TYPE( op(get( (l),s),get(r,s) ) )

public:

	template<typename TOP, typename TL>
	auto calculate(_Field<Expression<TOP, TL> > const & f, index_type s) const
	DECL_RET_TYPE((calculate_(f.op_,f.lhs,s)))

	template<typename TOP, typename TL, typename TR>
	auto calculate(_Field<Expression<TOP, TL, TR> > const & f,
			index_type s) const
			DECL_RET_TYPE((calculate_(f.op_,f.lhs,f.rhs,s)))

	template<typename TC, typename TD>
	auto calculate(_Field<TC, TD> const & f, index_type s) const
	DECL_RET_TYPE((f[s]))

	template<typename T, size_t ...N>
	nTuple<T, N...> const& calculate(nTuple<T, N...> const & v,
			index_type s) const
	{
		return v;
	}

	template<typename T>
	auto calculate(T const & v, index_type s) const
	DECL_RET_TYPE((get_value(v,s)))

};

}
// namespace simpla

#endif /* CORE_DIFF_GEOMETRY_DUMMY_MANIFOLD_H_ */
