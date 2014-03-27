/*
 * select.h
 *
 *  Created on: 2014年2月19日
 *      Author: salmon
 */

#ifndef SELECT_H_
#define SELECT_H_

#include "../utilities/log.h"
#include "../utilities/lua_state.h"
#include "../utilities/type_utilites.h"
namespace simpla
{
template<typename IT>
struct IteratorFilter
{
	static constexpr int MAX_CACHE_DEPTH = 12;

	typedef IT iterator;

	typedef typename iterator::value_type index_type;

	typedef IteratorFilter<iterator> this_type;

	typedef std::function<int(iterator, index_type*)> filter_type;

	filter_type filter_;

	iterator it_, ie_;

	index_type s_[MAX_CACHE_DEPTH];

	int cache_head_;

	int cache_tail_;

	IteratorFilter()
			: cache_head_(0), cache_tail_(0)
	{
	}
	IteratorFilter(iterator ib, iterator ie, filter_type filter)
			: it_(ib), ie_(ie), filter_(filter), cache_head_(0), cache_tail_(0)
	{
	}

	IteratorFilter(iterator ib, iterator ie)
			: it_(ib), ie_(ib), cache_head_(0), cache_tail_(0)
	{
		filter_ = []( index_type s, index_type* c)->int
		{
			c[0]=s;
			return 1;
		};
	}
	~IteratorFilter()
	{
	}
	this_type & operator ++()
	{
		++cache_head_;
		if (cache_head_ >= cache_tail_)
		{
			cache_tail_ = 0;
			cache_head_ = 0;
			while (it_ != ie_ && cache_tail_ == 0)
			{
				++it_;
				cache_tail_ = filter_(it_, s_);
			}

		}

		return *this;
	}

	bool operator==(this_type const & rhs)
	{
		return (it_ == rhs.it_) && (cache_head_ == rhs.cache_head_);
	}
	bool operator!=(this_type const & rhs)
	{
		return !(this->operator==(rhs));
	}

	index_type const & operator*() const
	{
		return s_[cache_head_];
	}

	index_type * operator ->()
	{
		return &s_[cache_head_];
	}
	index_type const* operator ->() const
	{
		return &s_[cache_head_];
	}

};

template<typename IT>
Range<IteratorFilter<IT>> Filter(IT ib, IT ie,
        std::function<
                int(IT,
                        typename std::remove_const<typename std::remove_reference<decltype(*std::declval<IT>())>::type>::type*)> const & filter)
{
	return Range<IteratorFilter<IT>>( { IteratorFilter<IT>(ib, ie, filter), IteratorFilter<IT>(ie) });
}

template<typename IT>
Range<IteratorFilter<IT>> Filter(IT ib, IT ie,
        std::function<bool(typename std::remove_reference<decltype(*std::declval<IT>())>::type)> const & filter)
{
	return Filter(ib, ie,
	        [&](IT s,typename std::remove_const<typename std::remove_reference<decltype(*std::declval<IT>())>::type>::type *c)->int
	        {	c[0]=*s; return filter(c[0])?1:0;});
}

template<typename TM>
Range<IteratorFilter<typename TM::iterator>> Filter(typename TM::iterator ib, typename TM::iterator ie, TM const &mesh,
        typename TM::coordinates_type const & x)
{
	UNIMPLEMENT;
}

template<typename TM>
Range<IteratorFilter<typename TM::iterator>> Filter(typename TM::iterator ib, typename TM::iterator ie, TM const & mesh,
        typename TM::coordinates_type const & v0, typename TM::coordinates_type const & v1)
{
	typedef typename TM::index_type index_type;
	return Filter(ib, ie,
	        [&]( typename TM::iterator s, index_type *c)->int
	        {
		        c[0]=*s;
		        auto x = mesh.GetCoordinates(*s);
		        return ((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
				        && (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0)) ? 1 : 0;
	        });

}

template<typename TM, int N>
Range<IteratorFilter<typename TM::iterator>> Filter(typename TM::iterator ib, typename TM::iterator ie, TM const & mesh,
        PointInPolygen checkPointsInPolygen)
{
	return Filter(ib, ie, [&](typename TM::iterator s,typename TM::index_type *c)->int
	{
		c[0]=*s;
		return (checkPointsInPolygen(mesh.GetCoordinates(c[0]) )) ? 1 : 0;
	});
}

/**
 *
 * @param mesh mesh
 * @param points  define Range
 *          if points.size() == 1 ,select Nearest Point
 *     else if points.size() == 2 ,select in the rectangle with  diagonal points[0] ~ points[1]
 *     else if points.size() >= 3 && Z<3
 *                    select points in a polyline on the Z-plane whose vertex are points
 *     else if points.size() >= 4 && Z>=3
 *                    select points in a closed surface
 *                    UNIMPLEMENTED
 *     else   illegal input
 *
 * @param fun
 * @param Z  Z==0    polyline on yz-plane
 *           Z==1    polyline on zx-plane,
 *           Z==2    polyline on xy-plane
 *           Z>=3
 */
template<typename TM, int N>
Range<IteratorFilter<typename TM::iterator>> Filter(typename TM::iterator ib, typename TM::iterator ie, TM const &mesh,
        std::vector<nTuple<N, Real>> const & points, unsigned int Z)
{
	Range<IteratorFilter<TM>> res;
	if (points.size() == 1)
	{

		typename TM::coordinates_type x = { 0, 0, 0 };

		for (int i = 0; i < N; ++i)
		{
			x[(i + Z + 1) % 3] = points[0][i];
		}

		res = Filter(ib, ie, mesh, x);
	}
	else if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
	{
		typename TM::coordinates_type v0 = { 0, 0, 0 };
		typename TM::coordinates_type v1 = { 0, 0, 0 };
		for (int i = 0; i < N; ++i)
		{
			v0[(i + Z + 1) % 3] = points[0][i];
			v1[(i + Z + 1) % 3] = points[1][i];
		}
		res = Filter(ib, ie, mesh, v0, v1);
	}
	else if (Z < 3 && points.size() > 2) //select points in polyline
	{
		return Filter(ib, ie, mesh, PointInPolygen(points, Z));
	}
	else
	{
		ERROR << "Illegal input";
	}
	return res;
}

template<typename TM>
Range<IteratorFilter<typename TM::iterator>> Filter(typename TM::iterator ib, typename TM::iterator ie, TM const &mesh,
        LuaObject const & dict)
{
	Range<IteratorFilter<TM>> res(mesh, ie);
	if (dict.is_table())
	{
		std::vector<typename TM::coordinates_type> points;

		dict.as(&points);

		res = Filter(ib, ie, mesh, points);

	}
	else if (dict.is_function())
	{

		res = Filter(ib, ie, [&](typename TM::iterator s,typename TM::index_type *c)->int
		{
			c[0]=*s;
			auto x = mesh.GetCoordinates(c[0]);
			return (dict(x[0], x[1], x[2]).template as<bool>()) ? 1 : 0;
		});

	}
	return res;

}
}
// namespace simpla

#endif /* SELECT_H_ */
