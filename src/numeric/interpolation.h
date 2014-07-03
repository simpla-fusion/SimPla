/*
 * interpolation.h
 *
 *  Created on: 2013年12月2日
 *      Author: salmon
 */

#ifndef INTERPOLATION_H_
#define INTERPOLATION_H_
#include "../utilities/log.h"
#include "../utilities/ntuple.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
namespace simpla
{

/**
 *
 * @defgroup Numeric
 * @ingroup Numeric
 * @{
 */
/*
 * @brief interpolation
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
	Interpolation(std::shared_ptr<container_type> y, Args && ...args) :
			data_(y), interpolate_op_(std::forward<Args >(args)...)
	{
	}

	template<typename ...Args>
	Interpolation(Args && ...args) :
			data_(std::shared_ptr<container_type>(new container_type)), interpolate_op_(std::forward<Args >(args)...)
	{
	}

	template<typename TC, typename ...Args>
	Interpolation(TC const &y, Args && ...args) :
			data_(std::shared_ptr<container_type>(new container_type(y))), interpolate_op_(std::forward<Args >(args)...)
	{
	}

	inline container_type & data()
	{
		return *data_;
	}
	inline container_type const& data() const
	{
//		EXCEPT(data_ != nullptr);
		return *data_;
	}

	~Interpolation()
	{
	}

	void swap(this_type & r)
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
		iterator jt = data_->upper_bound(x);
//		if (jt == data_->begin() || (++jt == data_->begin()))
//		{
//			LOGIC_ERROR << "key value out of range!";
//		}
		for (int s = 0; s < 2; ++s)
		{
			if (jt == data_->begin())
			{
				break;
			}
			--jt;
		}

		return jt;
	}

	value_type operator()(key_x_type const &x) const
	{
		return std::move(interpolate_op_.eval(*data_, find(x), x));
	}
	value_type eval(key_x_type const &x) const
	{
		return std::move(interpolate_op_.eval(*data_, find(x), x));
	}
	value_type diff(key_x_type const &x) const
	{
		return std::move(interpolate_op_.diff(*data_, find(x), x));
	}

};
/**
 * @ingroup Numeric
 *
 * @brief Linear Interpolation
 */
struct LinearInterpolation
{
	typedef LinearInterpolation this_type;

	void swap(this_type &r)
	{

	}

	template<typename container>
	inline typename container::mapped_type eval(container const &, typename container::iterator const &it,
	        typename container::key_type const &x) const
	{
		typedef typename container::mapped_type value_type;
		typename container::iterator next = it;
		++next;
		return it->second
		        + (static_cast<value_type>(x - it->first) / static_cast<value_type>(next->first - it->first))
		                * (next->second - it->second);

	}

	template<typename container>
	inline typename container::mapped_type diff(container const &, typename container::iterator const &it,
	        typename container::key_type const &x) const
	{
		typedef typename container::mapped_type value_type;
		typename container::iterator next = it;
		++next;
		return (next->second - it->second) / static_cast<value_type>(next->first - it->first);
	}
};

/**
 * @brief MultiDimesion Interpolation
 */

template<typename TInterpolator, typename TV, typename ...TX>
class MultiDimesionInterpolation: public TInterpolator
{

public:
	typedef MultiDimesionInterpolation<TInterpolator, TV> this_type;
	typedef TV value_type;
	enum
	{
		NDIMS = 2
	};
private:
	TInterpolator interpolate_op_;
	std::shared_ptr<value_type> data_;

public:

	template<typename ...Args>
	MultiDimesionInterpolation(std::shared_ptr<value_type> y, Args && ...args) :
			data_(y), interpolate_op_(std::forward<Args >(args)...)
	{
		Update();
	}

	template<typename ...Args>
	MultiDimesionInterpolation(Args && ...args) :
			data_(nullptr), interpolate_op_(std::forward<Args >(args)...)
	{
		Update();
	}

	virtual ~MultiDimesionInterpolation()
	{
	}

	virtual void swap(this_type & r)
	{
		interpolate_op_.swap(r.interpolate_op_);
		std::swap(data_, r.data_);

	}
	std::shared_ptr<value_type> data()
	{
		return data_;
	}
	const std::shared_ptr<value_type> data() const
	{
		return data_;
	}
	void Update()
	{

		interpolate_op_.Update();

		if (data_ == nullptr)
		{
			data_ = std::shared_ptr<value_type>(new value_type[interpolate_op_.GetNumberOfElement()]);
		}

	}

	template<typename ...TArgs>
	inline value_type operator()(TArgs const &... x) const
	{
		return std::move(interpolate_op_.eval(data_, std::forward<TArgs const &>(x)...));
	}

	inline value_type const &operator[](size_t s) const
	{
		return (&(*data_))[s];
	}
	inline value_type &operator[](size_t s)
	{
		return (&(*data_))[s];
	}

	template<typename ...TArgs>
	value_type eval(TArgs const &... x) const
	{
		return std::move(interpolate_op_.eval(data_.get(), std::forward<TArgs const &>(x)...));
	}

	template<typename ...TArgs>
	nTuple<NDIMS, value_type> diff(TArgs const &... x) const
	{
		return std::move(interpolate_op_.diff(data_.get(), std::forward<TArgs const &>(x)...));
	}

};
/**
 * @brief BiLinear Interpolation
 */
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
		for (int s = 0; s < NDIMS; ++s)
		{
			dims_[s] = 1;
		}
	}
	BiLinearInterpolation(nTuple<NDIMS, size_t> dims, nTuple<NDIMS, Real> const &xmin, nTuple<NDIMS, Real> const &xmax) :
			dims_(dims), xmin_(xmin), xmax_(xmax)
	{
		Update();
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
	inline void SetDefineDomain(nTuple<NDIMS, Real> const &xmin, nTuple<NDIMS, Real> const & xmax)
	{
		xmin_ = xmin;
		xmax_ = xmax;
	}

	inline void GetDefineDomain(nTuple<NDIMS, Real> *xmin, nTuple<NDIMS, Real> * xmax)
	{
		*xmin = xmin_;
		*xmax = xmax_;
	}

	size_t GetNumberOfElement() const
	{
		size_t res = 1;
		for (int i = 0; i < NDIMS; ++i)
		{
			res *= dims_[i];
		}
		return res;
	}

	inline void Update()
	{
		inv_dx_[0] = static_cast<Real>(dims_[0] - 1) / (xmax_[0] - xmin_[0]);
		inv_dx_[1] = static_cast<Real>(dims_[1] - 1) / (xmax_[1] - xmin_[1]);
	}

	template<typename TV, typename TX, typename ... Args>
	inline auto eval(TV const * v, TX x, TX y, Args const &...) const ->TV
	{
		x = (x - xmin_[0]) * inv_dx_[0];
		y = (y - xmin_[1]) * inv_dx_[1];

		Real ix, iy;
		Real rx = std::modf(x, &ix);
		Real ry = std::modf(y, &iy);

		size_t sx = 1;
		size_t sy = dims_[0];
		size_t s = static_cast<size_t>(ix) * sx + static_cast<size_t>(iy) * sy;

		return (

		(1.0 - rx) * (1.0 - ry) * v[s] +

		(1.0 - rx) * ry * v[s + sy] +

		rx * ry * v[s + sx + sy] +

		rx * (1.0 - ry) * v[s + sx]

		);

	}

	template<typename TV, typename TX>
	inline auto eval(TV const * v, TX const & x) const->decltype(eval(v, x[0], x[1]))
	{
		return eval(v, x[0], x[1]);
	}

	template<typename TV, typename TX, typename ... Args>
	inline nTuple<NDIMS, TV> diff(TV const * v, TX x, TX y, Args const &...) const
	{

		x = (x - xmin_[0]) * inv_dx_[0];
		y = (y - xmin_[1]) * inv_dx_[1];

		Real ix, iy;
		Real rx = std::modf(x, &ix);
		Real ry = std::modf(y, &iy);

		size_t sx = 1;
		size_t sy = dims_[0];
		size_t s = static_cast<size_t>(ix) * sx + static_cast<size_t>(iy) * sy;

		nTuple<NDIMS, TV> res = {

		(1.0 - ry) * (v[s + sx] - v[s]) + ry * (v[s + sx + sy] - v[s + sy]),

		(1.0 - rx) * (v[s + sy] - v[s]) + rx * (v[s + sx + sy] - v[s + sx]) };

		return std::move(res);

	}

	template<typename TV, int N, typename TX>
	inline auto diff(TV const & v, nTuple<N, TX> const & x) const
	DECL_RET_TYPE(std::move(diff(v, x[0], x[1])))

}
;
//! @}
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
//	void Update(container_type const &xy)
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
//		Update(xy);
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
