/*
 * interpolation.h
 *
 *  Created on: 2013年12月2日
 *      Author: salmon
 */

#ifndef INTERPOLATION_H_
#define INTERPOLATION_H_
#include "simpla_defs.h"
#include "utilities/log.h"
#include <vector>

namespace simpla
{

/**
 * @ref Numerical Recipes (C++ ,3rd 2007) Chap. 3
 *
 *
 */
struct LinearInterpolation;

template<typename TK, typename TY, typename TInterpolator = LinearInterpolation>
struct Interpolation: public std::map<TK, TY>
{

	TInterpolator interpolate_op_;
public:
	typedef Interpolation<TK, TY, TInterpolator> this_type;
	typedef std::map<TK, TY> container_type;
	typedef typename container_type::iterator iterator;
	typedef typename container_type::key_type key_type;
	typedef typename iterator::value_type value_type;

	Interpolation()
	{
	}
	template<typename ...Args>
	Interpolation(container_type const &xy, Args const & ...args) :
			container_type(xy), interpolate_op_(
					std::forward<Args const &>(args)...)
	{
	}
	template<typename ...Args>
	Interpolation(container_type &&xy, Args const & ...args) :
			container_type(xy), interpolate_op_(
					std::forward<Args const &>(args)...)
	{
	}
	virtual ~Interpolation()
	{
	}

	virtual void swap(this_type & r)
	{
		container_type::swap(r);
		interpolate_op_.swap(r.interpolate_op_);
	}

	/**
	 *
	 *
	 * @param x Given a value x,
	 * @return return a value j such that x is (insofar as possible) centered in the subrange
	 * xx[j..j+mm-1], where xx is the stored pointer. The values in xx must be monotonic, either
	 * increasing or decreasing. The returned value is not less than 0, nor greater than n-1.
	 */
	inline iterator find(key_type const & x) const
	{
		iterator jt = container_type::upper_bound(x);
		if (jt == container_type::begin() || (jt++ == container_type::begin()))
		{
			LOGIC_ERROR << "key value out of range!";
		}
		for (int s = 0; s < m / 2; ++s)
		{
			if (jt == container_type::begin())
			{
				break;
			}
			--jt;
		}

		return jt;
	}

	value_type operator()(key_type const &x) const
	{
		return std::move(interpolate_op_(*this, find(x), x));
	}

};

struct LinearInterpolation
{
	typedef LinearInterpolation this_type;

	void swap(this_type &r)
	{

	}

	template<typename container>
	inline typename container::mapped_type operator()(container const &,
			typename container::iterator const &it,
			typename container::key_type const &x)
	{
		typename container::iterator next = it++;
		return it->second
				+ ((x - it->first) / (next->first - it->first))
						* (next->second - it->second);

	}
};

class PolynomialInterpolation
{
	int m_;
public:

	typedef PolynomialInterpolation this_type;

	PolynomialInterpolation(int m) :
			m_(m)
	{
	}
	~PolynomialInterpolation(int m)
	{

	}

	void swap(this_type &r)
	{
	}

	template<typename container>
	inline typename container::mapped_type operator()(container const & xy,
			typename container::iterator const &it,
			typename container::key_type const &x) const
	{
		typedef typename container::iterator iterator;
		typedef typename container::key_type key_type;
		typedef typename container::mapped_type value_type;

		std::vector<key_type> xa;

		value_type y;

		std::vector<value_type> ya, c, d;

		key_type dif = abs(x - it->first);

		int ns = 0;

		for (int i = 0; i < m_; ++i)
		{
			key_type dift = abs(x - it->first);
			if (dift < dif)
			{
				ns = i;
				dif = dift;
			}
			xa.push_back(it->first);
			c.push_back(it->second);
			d.push_back(it->second);

			++it;

			if (it == xy.end())
				break;

		}

		y = xa[ns];

		for (int m = 1, m_end = c.size(); m < m_end; ++m)
		{
			for (int i = 0; i < m_end - m; ++i)
			{
				key_type ho = xa[i] - x;
				key_type hp = xa[i + m] - x;
				key_type w = c[i + 1] - d[i];
				key_type den = abs(ho - hp);
				if ((den) == 0.0)
					throw("Poly_interp error");

				den = w / den;

				d[i] = hp * den;

				c[i] = ho * den;
			}
			key_type dy = (2 * (ns + 1) < (m_ - m) ? c[ns + 1] : d[ns--]);
			y += dy;

		}
		return y;
	}
}
;
//
//template<typename TX, typename TY>
//class CubicSplineInterpolation
//{
//
//public:
//
//	typedef CubicSplineInterpolation<TX, TY> this_type;
//	typedef TX key_type;
//	typedef TY value_type;
//
//private:
//	std::vector<value_type> y2_;
//	value_type yp1_;
//	value_type yp2_;
//	void * data_;
//public:
//
//	CubicSplineInterpolation() :
//			data_(nullptr)
//	{
//	}
//
//	CubicSplineInterpolation(value_type const & yp1 = 1.0e99,
//			value_type const & yp2 = 1.0e99) :
//			yp1_(yp1), yp2_(yp2), data_(nullptr)
//	{
//
//	}
//
//	void swap(this_type &r)
//	{
//		y2_.swap(r.y2_);
//		std::swap(yp1_, r.yp1_);
//		std::swap(yp2_, r.yp2_);
//		std::swap(data_, r.data_);
//	}
//
//	/**
//	 * This routine stores an array y2[0..n-1] with second derivatives of the interpolating function
//	 at the tabulated points pointed to by xv, using function values pointed to by yv. If yp1 and/or
//	 ypn are equal to 1 1099 or larger, the routine is signaled to set the corresponding boundary
//	 condition for a natural spline, with zero second derivative on that boundary; otherwise, they are
//	 the values of the first derivatives at the endpoints.
//	 * @param xv
//	 * @param yv
//	 * @param yp1
//	 * @param ypn
//	 */
//	template<typename container_type>
//	void update(container_type const &xy)
//	{
//		if (static_cast<void const *>(&xy) == data_)
//			return;
//
//		data_ = static_cast<void const *>(&xy);
//
//		int i, k;
//		double p, qn, sig, un;
//		int n = y2_.size();
//		std::vector<value_type> u(n - 1);
//		if (yp1_ > 0.99e99)
//			y2_[0] = u[0] = 0.0;
//		else
//		{
//			y2_[0] = -0.5;
//			auto it1 = xy.begin();
//			auto it2 = it1++;
//			u[0] = (3.0 / (it2->first - it1->first))
//					* ((it2->second - it1->second) / (it2->first - it1->first)
//							- yp1_);
//		}
//		for (i = 1; i < n - 1; i++)
//		{
//			sig = (xv[i] - xv[i - 1]) / (xv[i + 1] - xv[i - 1]);
//			p = sig * y2_[i - 1] + 2.0;
//			y2_[i] = (sig - 1.0) / p;
//			u[i] = (yv[i + 1] - yv[i]) / (xv[i + 1] - xv[i])
//					- (yv[i] - yv[i - 1]) / (xv[i] - xv[i - 1]);
//			u[i] = (6.0 * u[i] / (xv[i + 1] - xv[i - 1]) - sig * u[i - 1]) / p;
//		}
//		if (ypn > 0.99e99)
//			qn = un = 0.0;
//		else
//		{
//			qn = 0.5;
//			un = (3.0 / (xv[n - 1] - xv[n - 2]))
//					* (ypn - (yv[n - 1] - yv[n - 2]) / (xv[n - 1] - xv[n - 2]));
//		}
//		y2_[n - 1] = (un - qn * u[n - 2]) / (qn * y2_[n - 2] + 1.0);
//		for (k = n - 2; k >= 0; k--)
//			y2_[k] = y2_[k] * y2_[k + 1] + u[k];
//	}
//
//	/**
//	 * Given a value x, and using pointers to data xx and yy, and the stored vector of second derivatives
//	 y2, this routine returns the cubic spline interpolated value y.
//	 * @param jl
//	 * @param x
//	 * @return
//	 */
//
//	template<typename container>
//	inline typename container::mapped_type operator()(container const & xy,
//			typename container::iterator const &it,
//			typename container::key_type const &x)
//	{
//		update(xy);
//		const_cast<this_type const *>(this)->operator()(xy, it, x);
//	}
//
//	template<typename container>
//	inline typename container::mapped_type operator()(container const & xy,
//			typename container::iterator const &it,
//			typename container::key_type const &x) const
//	{
//		typedef typename container::iterator iterator;
//		typedef typename container::key_type key_type;
//		typedef typename container::mapped_type value_type;
//
//		key_type klo = jl, khi = jl + 1;
//		value_type y;
//		key_type h, b, a;
//		h = xy[khi] - xy[klo];
//		if (h == 0.0)
//			throw("Bad input to routine splint");
//		a = (xy[khi] - x) / h;
//
//		b = (x - xy[klo]) / h;
//		y = a * xy[klo] + b * xy[khi]
//				+ ((a * a * a - a) * y2_[klo] + (b * b * b - b) * y2_[khi])
//						* (h * h) / 6.0;
//		return y;
//
//	}
//
//}
//;
}
// namespace simpla

#endif /* INTERPOLATION_H_ */
