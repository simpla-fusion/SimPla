/**
 * find_root.h
 *
 * \date 2014年7月6日
 * \author salmon
 */

#ifndef FIND_ROOT_H_
#define FIND_ROOT_H_
#include <tuple>
namespace simpla
{
//!  \ingroup Numeric
//! @{

/**
 *
 * using Newton's method find root
 *
 * @param x0 \f$x_{0}\f$
 * @param x1 \f$x_{1}\f$
 * @param f function
 * @param y_m \f$y_m\f$
 * @param resolution
 * @param max_iterator_num
 * @return
 */
template<typename TX, typename TY>
std::tuple<bool, TX> find_root(TX x0, TX x1, std::function<TY(TX const&)> const &f, TY const & y_m, Real resolution =
        0.001, size_t max_iterator_num = 10000)
{
	// @todo need change to Newton method!!!
	resolution *= abs(x1 - x0);

	bool success = false;

	TX x;

	TY y0 = f(x0);
	TY y1 = f(x1);
	size_t count = 0;
	do
	{
		++count;
		TX x = (x0 + x1) / 2;

		TY y = f(x);

		if (y < y_m == y0 < y_m)
		{
			y0 = y;
			x0 = x;
		}
		else if (y < y_m == y1 < y_m)
		{
			y1 = y;
			x1 = x;
		}

		success = (y0 < y_m) != (y1 < y_m);

		if (!success)
			break;

	} while (abs(x1 - x0) > resolution && count < max_iterator_num);

	return std::forward_as_tuple(success && (count < max_iterator_num), std::move(x0));

}

/**
 *
 * using mid-point(binary search) method find root
 *
 * @param x0 \f$x_{0}\f$
 * @param x1 \f$x_{1}\f$
 * @param f function
 * @param resolution
 * @param max_iterator_num
 * @return
 */
template<typename TX>
std::tuple<bool, TX> find_root(TX x0, TX x1, std::function<bool(TX const&)> const &f, Real resolution = 0.001,
        size_t max_iterator_num = 10000)
{
	resolution *= abs(x1 - x0);

	bool success = false;

	bool y0 = f(x0);
	bool y1 = f(x1);
	size_t count = 0;
	do
	{
		++count;
		TX x = (x0 + x1) / 2;

		bool y = f(x);

		if (y == y0)
		{
			y0 = y;
			x0 = x;
		}
		else if (y == y1)
		{
			y1 = y;
			x1 = x;
		}

		success = y0 != y1;

		if (!success)
			break;

	} while (abs(x1 - x0) > resolution && count < max_iterator_num);

	return std::forward_as_tuple(success && (count < max_iterator_num), std::move(x0));
}

//!  @}
}// namespace simpla

#endif /* FIND_ROOT_H_ */
