/*
 * interpolation.h
 *
 *  Created on: 2013年12月2日
 *      Author: salmon
 */

#ifndef INTERPOLATION_H_
#define INTERPOLATION_H_
#include "../simpla_defs.h"
#include "../utilities/log.h"
#include "../fetl/ntuple.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
namespace simpla
{

/**
 * @ref Numerical Recipes (C++ ,3rd 2007) Chap. 3
 *
 *
 */

template<typename TInterpolator, typename TV, typename TX>
class Interpolation
{

public:
	typedef Interpolation<TInterpolator, TV, TX> this_type;
	typedef std::map<TX, TV> container_type;
	typedef typename container_type::iterator iterator;
	typedef TX key_x_type;
	typedef TV value_type;

private:
	TInterpolator interpolate_op_;
	std::shared_ptr<container_type> data_;
public:

	template<typename ...Args>
	Interpolation(std::shared_ptr<container_type> y, Args const & ...args) :
			data_(y), interpolate_op_(std::forward<Args const &>(args)...)
	{
	}

	template<typename ...Args>
	Interpolation(Args const & ...args) :
			Interpolation(std::shared_ptr<container_type>(new container_type()),
					std::forward<Args const &>(args)...)
	{
	}

	template<typename TC, typename ...Args>
	Interpolation(TC const &y, Args const & ...args) :
			Interpolation(
					std::shared_ptr<container_type>(new container_type(y)),
					std::forward<Args const &>(args)...)
	{
	}

	inline container_type & data()
	{
		return *data_;
	}
	inline container_type const& data() const
	{
		EXCEPT(data_ != nullptr);
		return *data_;
	}

	virtual ~Interpolation()
	{
	}

	virtual void swap(this_type & r)
	{
		std::swap(data_, r.data_);
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
	inline iterator find(key_x_type const & x) const
	{
		iterator jt = container_type::upper_bound(x);
		if (jt == container_type::begin() || (jt++ == container_type::begin()))
		{
			LOGIC_ERROR << "key value out of range!";
		}
		for (int s = 0; s < interpolate_op_.m_ / 2; ++s)
		{
			if (jt == container_type::begin())
			{
				break;
			}
			--jt;
		}

		return jt;
	}

	value_type operator()(key_x_type const &x) const
	{
		return std::move(interpolate_op_(*data_, find(x), x));
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
		typedef typename container::mapped_type value_type;
		typename container::iterator next = it++;
		return it->second
				+ (static_cast<value_type>(x - it->first)
						/ static_cast<value_type>(next->first - it->first))
						* (next->second - it->second);

	}
};
//
//class PolynomialInterpolation
//{
//	int m_;
//public:
//
//	typedef PolynomialInterpolation this_type;
//
//	PolynomialInterpolation(int m) :
//			m_(m)
//	{
//	}
//	~PolynomialInterpolation(int m)
//	{
//
//	}
//
//	void swap(this_type &r)
//	{
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
//		std::vector<key_type> xa;
//
//		value_type y;
//
//		std::vector<value_type> ya, c, d;
//
//		key_type dif = abs(x - it->first);
//
//		int ns = 0;
//
//		for (int i = 0; i < m_; ++i)
//		{
//			key_type dift = abs(x - it->first);
//			if (dift < dif)
//			{
//				ns = i;
//				dif = dift;
//			}
//			xa.push_back(it->first);
//			c.push_back(it->second);
//			d.push_back(it->second);
//
//			++it;
//
//			if (it == xy.end())
//				break;
//
//		}
//
//		y = xa[ns];
//
//		for (int m = 1, m_end = c.size(); m < m_end; ++m)
//		{
//			for (int i = 0; i < m_end - m; ++i)
//			{
//				key_type ho = xa[i] - x;
//				key_type hp = xa[i + m] - x;
//				value_type w = c[i + 1] - d[i];
//				value_type den = abs(static_cast<value_type>(ho - hp));
//				if ((den) == 0.0)
//					throw("Poly_interp error");
//
//				den = w / den;
//
//				d[i] = hp * den;
//
//				c[i] = ho * den;
//			}
//			y += (2 * (ns + 1) < (m_ - m) ? c[ns + 1] : d[ns--]);
//
//		}
//		return y;
//	}
//}
//;
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
template<typename TInterpolator, typename TV, typename ...TX>
class MultiDimesionInterpolation: public TInterpolator
{

public:
	typedef MultiDimesionInterpolation<TInterpolator, TV> this_type;
	typedef std::vector<TV> container_type;
	typedef typename container_type::iterator iterator;
	typedef TV value_type;
	enum
	{
		NDIMS = 2
	};
private:
	TInterpolator interpolate_op_;
	std::shared_ptr<container_type> data_;

public:

	template<typename ...Args>
	MultiDimesionInterpolation(std::shared_ptr<container_type> y,
			Args const & ...args) :
			data_(y), interpolate_op_(std::forward<Args const &>(args)...)
	{
		update();
	}

	template<typename ...Args>
	MultiDimesionInterpolation(Args const & ...args) :
			data_(std::shared_ptr<container_type>(new container_type())), interpolate_op_(
					std::forward<Args const &>(args)...)
	{
	}

	template<typename ...Args>
	MultiDimesionInterpolation(container_type const &y, Args const & ...args) :
			MultiDimesionInterpolation(
					std::shared_ptr<container_type>(new container_type(y)),
					std::forward<Args const &>(args)...)
	{
	}

	virtual ~MultiDimesionInterpolation()
	{
	}

	virtual void swap(this_type & r)
	{
		interpolate_op_.swap(r.interpolate_op_);
		std::swap(data_, r.data_);

	}
	inline container_type & data()
	{
		return *data_;
	}
	inline container_type const& data() const
	{
		return *data_;
	}
	void update()
	{
		interpolate_op_.update();

		if (data_ == nullptr)
		{
			data_ = std::shared_ptr<container_type>(new container_type());
		}

	}

	template<typename ...TArgs>
	inline value_type operator()(TArgs const &... x) const
	{
		return std::move(
				interpolate_op_.eval(*data_, std::forward<TArgs>(x)...));
	}
};

class BiLinearInterpolation
{
public:
	typedef BiLinearInterpolation this_type;
	enum
	{
		NDIMS = 2
	};

private:

	nTuple<NDIMS, size_t> dims_;
	nTuple<NDIMS, Real> xmin_, xmax_, inv_dx_;
public:
	BiLinearInterpolation()
	{

	}
	BiLinearInterpolation(nTuple<NDIMS, size_t> dims,
			nTuple<NDIMS, Real> const &xmin, nTuple<NDIMS, Real> const &xmax) :
			dims_(dims), xmin_(xmin), xmax_(xmax)
	{

	}
	~BiLinearInterpolation()
	{

	}
	void swap(this_type &r)
	{
		std::swap(dims_, r.dims_);
		std::swap(xmin_, r.xmin_);
		std::swap(xmax_, r.xmax_);
		std::swap(inv_dx_, r.inv_dx_);
	}
	inline void SetDimension(nTuple<NDIMS, size_t> const &dims)
	{
		dims_ = dims;
	}
	inline void GetDimension(nTuple<NDIMS, size_t> * dims) const
	{
		*dims = dims_;
	}
	inline void SetDefineDomain(nTuple<NDIMS, Real> const &xmin,
			nTuple<NDIMS, Real> const & xmax)
	{
		xmin_ = xmin;
		xmax_ = xmax;
	}

	inline void GetDefineDomain(nTuple<NDIMS, Real> *xmin,
			nTuple<NDIMS, Real> * xmax)
	{
		*xmin = xmin_;
		*xmax = xmax_;
	}

	inline void update()
	{
		inv_dx_[0] = static_cast<Real>(dims_[0] - 1) / (xmax_[0] - xmin_[0]);

		inv_dx_[1] = static_cast<Real>(dims_[1] - 1) / (xmax_[1] - xmin_[1]);
	}

	template<typename TV, typename TX, typename ... Args>
	inline auto eval(TV const & v, TX const & x, TX const & y,
			Args const &...)->decltype(v[0])
	{
		x = NormalizeX(x);
		y = NormalizeX(y);

		Real rx = std::ceil(x);
		Real ry = std::ceil(y);
		size_t ix = static_cast<size_t>(std::floor(x));
		size_t iy = static_cast<size_t>(std::floor(y));
		size_t ny = dims_[1];
		return (

		(1.0 - rx) * (1.0 - ry) * v[ix * ny + iy] +

		(1.0 - rx) * ry * v[ix * ny + iy + 1] +

		rx * ry * v[(ix + 1) * ny + iy + 1] +

		rx * (1.0 - ry) * v[(ix + 1) * ny + iy]

		);

	}

	template<typename TV, typename TX, typename ... Args>
	inline auto eval(TV const & v, TX const & x, TX const & y,
			Args const &...)->decltype(v[0][0])
	{

		x = NormalizeX(x);
		y = NormalizeX(y);

		Real rx = std::ceil(x);
		Real ry = std::ceil(y);
		size_t ix = static_cast<size_t>(std::floor(x));
		size_t iy = static_cast<size_t>(std::floor(y));

		return (

		(1.0 - rx) * (1.0 - ry) * v[ix][iy] +

		(1.0 - rx) * ry * v[ix][iy + 1] +

		rx * ry * v[ix + 1][iy + 1] +

		rx * (1.0 - ry) * v[ix + 1][iy]

		);

	}

	template<typename T>
	inline Real NormalizeX(T x) const
	{
		return (x - xmin_[0]) * inv_dx_[0];
	}
	template<typename T>
	inline Real NormalizeY(T y) const
	{
		return (y - xmin_[1]) * inv_dx_[1];
	}

};
}
// namespace simpla

#endif /* INTERPOLATION_H_ */
