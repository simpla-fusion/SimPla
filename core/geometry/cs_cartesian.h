/*
 * cs_cartesian.h
 *
 *  Created on: 2015年6月14日
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CS_CARTESIAN_H_
#define CORE_GEOMETRY_CS_CARTESIAN_H_
#include "../gtl/macro.h"
#include "../gtl/type_cast.h"
#include "coordinate_system.h"
namespace simpla
{

namespace geometry
{

namespace st = simpla::traits;
namespace gt = geometry::traits;

template<typename, typename > struct map;

template<typename > struct mertic;

template<size_t ZAXIS0, size_t ZAXIS1>
struct map<coordinate_system::Cartesian<3, ZAXIS0>,
		coordinate_system::Cartesian<3, ZAXIS1> >
{

	static constexpr size_t CartesianZAxis0 = (ZAXIS0) % 3;
	static constexpr size_t CartesianYAxis0 = (CartesianZAxis0 + 2) % 3;
	static constexpr size_t CartesianXAxis0 = (CartesianZAxis0 + 1) % 3;
	typedef gt::point_t<coordinate_system::Cartesian<3, ZAXIS0> > point_t0;
	typedef gt::vector_t<coordinate_system::Cartesian<3, ZAXIS0> > vector_t0;
	typedef gt::covector_t<coordinate_system::Cartesian<3, ZAXIS0> > covector_t0;

	static constexpr size_t CartesianZAxis1 = (ZAXIS1) % 3;
	static constexpr size_t CartesianYAxis1 = (CartesianZAxis1 + 2) % 3;
	static constexpr size_t CartesianXAxis1 = (CartesianZAxis1 + 1) % 3;
	typedef gt::point_t<coordinate_system::Cartesian<3, ZAXIS1> > point_t1;
	typedef gt::vector_t<coordinate_system::Cartesian<3, ZAXIS1> > vector_t1;
	typedef gt::covector_t<coordinate_system::Cartesian<3, ZAXIS1> > covector_t1;

	static point_t1 eval(point_t0 const & x)
	{
		/**
		 *  @note
		 * coordinates transforam
		 *
		 *  \f{eqnarray*}{
		 *		x & = & r\cos\phi\\
			 *		y & = & r\sin\phi\\
			 *		z & = & Z
		 *  \f}
		 *
		 */
		point_t1 y;

		st::get<CartesianXAxis1>(y) = st::get<CartesianXAxis0>(x);

		st::get<CartesianYAxis1>(y) = st::get<CartesianYAxis0>(x);

		st::get<CartesianZAxis1>(y) = st::get<CartesianZAxis0>(x);

		return std::move(y);
	}
	point_t1 operator()(point_t0 const & x) const
	{
		return eval(x);
	}

	template<typename TFun>
	auto pull_back(point_t0 const & x0, TFun const &fun)
	DECL_RET_TYPE ((fun(map (x0))))

	template<typename TRect>
	TRect pull_back(point_t0 const & x0,
			std::function<TRect(point_t0 const &)> const &fun)
	{
		return fun(map(x0));
	}

	/**
	 *
	 *   push_forward vector from Cylindrical  to Cartesian
	 * @param R  \f$ v=v_{r}\partial_{r}+v_{Z}\partial_{Z}+v_{\theta}/r\partial_{\theta} \f$
	 * @param CartesianZAxis
	 * @return  \f$ \left(x,y,z\right),u=u_{x}\partial_{x}+u_{y}\partial_{y}+u_{z}\partial_{z} \f$
	 *
	 */
	vector_t1 push_forward(point_t0 const & x0, vector_t0 const &v)
	{

		vector_t1 u;

		st::get<CartesianXAxis1>(u) = st::get<CartesianXAxis0>(v);
		st::get<CartesianYAxis1>(u) = st::get<CartesianYAxis0>(v);
		st::get<CartesianZAxis1>(u) = st::get<CartesianZAxis0>(v);

		return std::move(u);
	}

};
template<size_t ICARTESIAN_ZAXIS>
struct mertic<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> >
{

	static constexpr size_t CartesianZAxis = (ICARTESIAN_ZAXIS) % 3;
	static constexpr size_t CartesianYAxis = (CartesianZAxis + 2) % 3;
	static constexpr size_t CartesianXAxis = (CartesianZAxis + 1) % 3;
	typedef gt::point_t<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> > point_t;
	typedef gt::vector_t<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> > vector_t;
	typedef gt::covector_t<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> > covector_t;

	typedef nTuple<Real, 3> delta_t;

	template<size_t DI>
	static constexpr Real dl(point_t const & x0, delta_t const & delta)
	{
		return st::get<DI>(delta);
	}

public:
	template<typename ...Others>
	static constexpr Real volume(size_t node_id, point_t const & x0,
			delta_t delta, Others && ...)
	{

		return (((node_id >> CartesianXAxis) & 1UL)
				* (dl<CartesianXAxis>(x0, delta)
						* st::get<CartesianXAxis>(delta) - 1) + 1.0)

				* (((node_id >> CartesianYAxis) & 1UL)
						* (dl<CartesianYAxis>(x0, delta)
								* st::get<CartesianYAxis>(delta) - 1) + 1.0)

				* (((node_id >> CartesianZAxis) & 1UL)
						* (dl<CartesianZAxis>(x0, delta)
								* st::get<CartesianZAxis>(delta) - 1) + 1.0)

		;
	}

	template<typename ...Others>
	static constexpr Real dual_volume(size_t node_id, Others && ...others)
	{
		return volume(7UL & (~node_id), std::forward<Others>(others)...);
	}

}
;
}  // namespace geometry
namespace traits
{

template<size_t NDIMS, size_t ICARTESIAN_ZAXIS>
struct type_id<
		geometry::coordinate_system::Cartesian<NDIMS, ICARTESIAN_ZAXIS> >
{
	static std::string name()
	{
		return "Cartesian<" + simpla::type_cast<std::string>(NDIMS) + ","
				+ simpla::type_cast<std::string>(ICARTESIAN_ZAXIS) + ">";
	}
};

}  // namespace traits
}  // namespace simpla

#endif /* CORE_GEOMETRY_CS_CARTESIAN_H_ */
