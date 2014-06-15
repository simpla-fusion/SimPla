/*
 * select.h
 *
 *  Created on: 2014年2月19日
 *      Author: salmon
 */

#ifndef SELECT_H_
#define SELECT_H_

#include "../utilities/log.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/utilities.h"

#include "pointinpolygen.h"

namespace std
{
template<typename TI> struct iterator_traits;
}  // namespace std
namespace simpla
{

//template<typename TRange>
//class RangeFilter
//{
//public:
//
//	typedef RangeFilter<TRange> this_type;
//
//	typedef TRange base_range;
//
//	typedef typename base_range::iterator base_iterator;
//
//	typedef typename base_iterator::value_type value_type;
//
//	typedef std::function<bool(base_iterator)> filter_type;
//
//private:
//	base_range range_;
//	filter_type filter_;
//public:
//
//	RangeFilter(base_range range, filter_type filter)
//			: range_(range), filter_(filter)
//	{
//	}
//
//	RangeFilter()
//			: range_(base_range())
//	{
//	}
//	~RangeFilter()
//	{
//	}
//
//	struct iterator
//	{
//		static constexpr int MAX_CACHE_DEPTH = 12;
//
//		filter_type filter_;
//
//		base_iterator it_, ie_;
//
//		iterator()
//		{
//		}
//
//		iterator(base_iterator ib, base_iterator ie, filter_type filter)
//				: it_(ib), ie_(ie), filter_(filter)
//		{
//			this->operator ++();
//		}
//		iterator(base_iterator it)
//				: it_(it), ie_(it)
//		{
//		}
//		iterator(iterator const &) = default;
//
////	iterator(iterator &&) = default;
//
//		~iterator()
//		{
//		}
//
//		bool operator==(iterator const & rhs)
//		{
//			return (it_ == rhs.it_);
//		}
//		bool operator!=(iterator const & rhs)
//		{
//			return !(this->operator==(rhs));
//		}
//		iterator & operator ++()
//		{
//
////			++cache_head_;
////			if (cache_head_ >= cache_tail_)
////			{
////				cache_tail_ = 0;
////				cache_head_ = 0;
////				while (it_ != ie_)
////				{
////					++it_;
////					cache_tail_ = filter_(it_, s_);
////					if (cache_tail_ > 0)
////						break;
////				}
////			}
//
//			for (++it_; it_ != ie_ && !filter_(it_); ++it_)
//			{
//			}
//			return *this;
//		}
//		this_type operator ++(int) const
//		{
//			this_type res(*this);
//			++res;
//			return res;
//		}
//
//		value_type const & operator*() const
//		{
//			return it_.operator*();
//		}
//
//		value_type * operator ->()
//		{
//			return it_.operator->();
//		}
//		value_type const* operator ->() const
//		{
//			return it_.operator->();
//		}
//
//	};
//	bool empty()
//	{
//		return begin() == end();
//	}
//
//	iterator begin() const
//	{
//		return iterator(range_.begin(), range_.end(), filter_);
//	}
//	iterator end() const
//	{
//		return iterator(range_.end());
//	}
//	this_type split(size_t num, size_t id)
//	{
//		return this_type(range_.split(num, id), filter_);
//	}
//
//};

template<typename TM>
Range<FilterIterator<std::function<bool(typename TM::iterator::value_type const &)>, typename TM::iterator>> Filter(
        typename TM::range_type range, TM const &mesh, nTuple<3, Real> x)
{
	auto dest = mesh.CoordinatesGlobalToLocal(&x);

	std::function<bool(typename TM::iterator::value_type const &)> pred =
	        [dest,&mesh](typename TM::iterator::value_type const &s )->bool
	        {
		        return mesh.GetCellIndex(s)==dest;
	        };

	return make_filter_range(pred, range);
}

template<typename TM>
Range<FilterIterator<std::function<bool(typename TM::iterator::value_type const &)>, typename TM::iterator>> Filter(
        typename TM::range_type range, TM const & mesh, typename TM::coordinates_type v0,
        typename TM::coordinates_type v1)
{
	std::function<bool(typename TM::iterator::value_type const &)> pred =
	        [v0,v1,&mesh]( typename TM::iterator::value_type const &s )->bool
	        {
		        auto x = mesh.GetCoordinates(s);
		        return ((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
				        && (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0));
	        };
	return make_filter_range(pred, range);
}

template<typename TM>
Range<FilterIterator<std::function<bool(typename TM::iterator::value_type const &)>, typename TM::iterator>> Filter(
        typename TM::range_type range, TM const & mesh, PointInPolygen checkPointsInPolygen)
{
	std::function<bool(typename TM::iterator::value_type const &)> pred =
	        [ checkPointsInPolygen,&mesh ](typename TM::iterator::value_type const &s )->bool
	        {	return (checkPointsInPolygen(mesh.GetCoordinates(s) ));};

	return make_filter_range(pred, range);

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
template<typename TR, typename TM, int N>
Range<FilterIterator<std::function<bool(typename TM::iterator::value_type const &)>, typename TM::iterator>> Filter(
        TR range, TM const &mesh, std::vector<nTuple<N, Real>> const & points, unsigned int Z = 2)
{
	Range<FilterIterator<std::function<bool(typename TR::iterator::value_type const &)>, typename TR::iterator>> res;

	if (points.size() == 1)
	{

		typename TM::coordinates_type x = { 0, 0, 0 };

		for (int i = 0; i < N; ++i)
		{
			x[(i + Z + 1) % 3] = points[0][i];
		}
		res = Filter(range, mesh, x);
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
		res = Filter(range, mesh, v0, v1);
	}
	else if (Z < 3 && points.size() > 2) //select points in polyline
	{
		return Filter(range, mesh, PointInPolygen(points, Z));
	}
	else
	{
		CHECK("WWWW");
		PARSER_ERROR("too less points " + ToString(points.size()));
	}
	return res;
}

template<typename TRange, typename TDict, typename TM>
Range<FilterIterator<std::function<bool(typename TRange::iterator::value_type const &)>, typename TRange::iterator>> Filter(
        TRange const & range, TM const & mesh, TDict const & dict)
{

	if (!dict)
	{
		CHECK("WWWW1");
		PARSER_ERROR("Empty configure information! ");
	}

	typedef typename TRange::iterator::value_type value_type;

	Range<FilterIterator<std::function<bool(value_type const &)>, typename TRange::iterator>> res;

	if (dict["Type"].template as<std::string>() == "Range" && dict["Points"].is_table())
	{
		std::vector<typename TM::coordinates_type> points;

		dict["Points"].as(&points);

		res = Filter(range, mesh, points);

	}
	else if (dict.is_function())
	{
		std::function<bool(value_type const &)> pred = [dict, &mesh]( value_type const & s )->bool
		{
			return (dict( mesh.GetCoordinates( s)).template as<bool>());
		};

		res = make_filter_range(pred, range);

	}
	return res;

}

} // namespace simpla

#endif /* SELECT_H_ */
