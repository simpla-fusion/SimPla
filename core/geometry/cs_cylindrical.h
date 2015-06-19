/**
 * @file cs_cylindrical.h
 *
 *  Created on: 2015年6月13日
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CS_CYLINDRICAL_H_
#define CORE_GEOMETRY_CS_CYLINDRICAL_H_
#include "../gtl/macro.h"
#include "../gtl/type_traits.h"
#include "coordinate_system.h"

namespace simpla
{

namespace geometry
{

namespace st = simpla::traits;
namespace gt = simpla::geometry::traits;

template<typename, typename > struct map;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
struct map<coordinate_system::Cylindrical<IPhiAxis>,
		coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> >
{

	typedef gt::point_t<coordinate_system::Cylindrical<IPhiAxis> > point_t0;
	typedef gt::vector_t<coordinate_system::Cylindrical<IPhiAxis> > vector_t0;
	typedef gt::covector_t<coordinate_system::Cylindrical<IPhiAxis> > covector_t0;

	static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
	static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
	static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;

	typedef gt::point_t<coordinate_system::Cartesian<3, CARTESIAN_XAXIS> > point_t1;
	typedef gt::vector_t<coordinate_system::Cartesian<3, CARTESIAN_XAXIS> > vector_t1;
	typedef gt::covector_t<coordinate_system::Cartesian<3, CARTESIAN_XAXIS> > covector_t1;

	static constexpr size_t CartesianZAxis = (ICARTESIAN_ZAXIS) % 3;
	static constexpr size_t CartesianYAxis = (CartesianZAxis + 2) % 3;
	static constexpr size_t CartesianXAxis = (CartesianZAxis + 1) % 3;

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

		st::get<CartesianXAxis>(y) = st::get<CylindricalRAxis>(x)
				* std::cos(st::get<CylindricalPhiAxis>(x));

		st::get<CartesianYAxis>(y) = st::get<CylindricalRAxis>(x)
				* std::sin(st::get<CylindricalPhiAxis>(x));

		st::get<CartesianZAxis>(y) = st::get<CylindricalZAxis>(x);

		return std::move(y);
	}
	point_t1 operator()(point_t0 const & x) const
	{
		return eval(x);
	}

	template<typename TFun>
	auto pull_back(point_t0 const & x0, TFun const &fun)
	DECL_RET_TYPE((fun(map(x0))))

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
	vector_t1 push_forward(point_t0 const & x0, vector_t0 const &v0)
	{

		Real cos_phi = std::cos(st::get<CylindricalPhiAxis>(x0));
		Real sin_phi = std::cos(st::get<CylindricalPhiAxis>(x0));
		Real r = st::get<CylindricalRAxis>(x0);

		Real vr = st::get<CylindricalRAxis>(v0);
		Real vphi = st::get<CylindricalPhiAxis>(v0);

		vector_t1 u;

		st::get<CartesianXAxis>(u) = vr * cos_phi - vphi * r * sin_phi;
		st::get<CartesianYAxis>(u) = vr * sin_phi + vphi * r * cos_phi;
		st::get<CartesianZAxis>(u) = st::get<CylindricalZAxis>(v0);

		return std::move(u);
	}

};

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cylindrical<IPhiAxis>,
		coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> >::CylindricalRAxis;
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cylindrical<IPhiAxis>,
		coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> >::CylindricalZAxis;
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cylindrical<IPhiAxis>,
		coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> >::CylindricalPhiAxis;
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cylindrical<IPhiAxis>,
		coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> >::CartesianXAxis;
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cylindrical<IPhiAxis>,
		coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> >::CartesianYAxis;
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cylindrical<IPhiAxis>,
		coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS> >::CartesianZAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
struct map<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS>,
		coordinate_system::Cylindrical<IPhiAxis> >
{

	typedef gt::point_t<coordinate_system::Cylindrical<IPhiAxis> > point_t1;
	typedef gt::vector_t<coordinate_system::Cylindrical<IPhiAxis> > vector_t1;
	typedef gt::covector_t<coordinate_system::Cylindrical<IPhiAxis> > covector_t1;

	static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
	static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
	static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;

	typedef gt::point_t<coordinate_system::Cartesian<3, CARTESIAN_XAXIS> > point_t0;
	typedef gt::vector_t<coordinate_system::Cartesian<3, CARTESIAN_XAXIS> > vector_t0;
	typedef gt::covector_t<coordinate_system::Cartesian<3, CARTESIAN_XAXIS> > covector_t0;

	static constexpr size_t CartesianZAxis = (ICARTESIAN_ZAXIS) % 3;
	static constexpr size_t CartesianYAxis = (CartesianZAxis + 2) % 3;
	static constexpr size_t CartesianXAxis = (CartesianZAxis + 1) % 3;

	static point_t1 eval(point_t0 const & x)
	{
		point_t1 y;
		/**
		 *  @note
		 *  coordinates transforam
		 *  \f{eqnarray*}{
		 *		r&=&\sqrt{x^{2}+y^{2}}\\
		 *		Z&=&z\\
		 *		\phi&=&\arg\left(y,x\right)
		 *  \f}
		 *
		 */
		st::get<CylindricalZAxis>(y) = st::get<CartesianYAxis>(x);
		st::get<CylindricalRAxis>(y) = std::sqrt(
				st::get<CartesianXAxis>(x) * st::get<CartesianXAxis>(x)
						+ st::get<CartesianZAxis>(x)
								* st::get<CartesianZAxis>(x));
		st::get<CylindricalPhiAxis>(y) = std::atan2(st::get<CartesianZAxis>(x),
				st::get<CartesianXAxis>(x));

		return std::move(y);

	}

	point_t1 operator()(point_t0 const & x) const
	{
		return eval(x);
	}
	template<typename TFun>
	auto pull_back(point_t0 const & x0, TFun const &fun)
	DECL_RET_TYPE((fun(map(x0))))

	template<typename TRect>
	TRect pull_back(point_t0 const & x0,
			std::function<TRect(point_t0 const &)> const &fun)
	{
		return fun(map(x0));
	}

	/**
	 *
	 *   push_forward vector from Cartesian to Cylindrical
	 *
	 *
	 * \verbatim
	 *
	 *     theta   y   r
	 *          \  |  /
	 *           \ | /
	 *            \|/------x
	 *          y  /
	 *          | /
	 *          |/)theta
	 *          0------x
	 *
	 * \endverbatim
	 *
	 * @param Z  \f$ \left(x,y,z\right),u=u_{x}\partial_{x}+u_{y}\partial_{y}+u_{z}\partial_{z} \f$
	 * @return  \f$ v=v_{r}\partial_{r}+v_{Z}\partial_{Z}+v_{\theta}/r\partial_{\theta} \f$
	 *
	 */
	vector_t1 push_forward(point_t0 const & x0, vector_t0 const &v0)
	{

		point_t1 y;

		y = map(x0);

		Real cos_phi = std::cos(st::get<CylindricalPhiAxis>(x0));
		Real sin_phi = std::cos(st::get<CylindricalPhiAxis>(x0));

		Real r = st::get<CylindricalRAxis>(x0);

		vector_t1 u;

		Real vx = st::get<CartesianXAxis>(v0);
		Real vy = st::get<CartesianYAxis>(v0);

		st::get<CylindricalPhiAxis>(u) = (-vx * sin_phi + vy * cos_phi) / r;

		st::get<CylindricalRAxis>(u) = vx * cos_phi + vy * sin_phi;

		st::get<CylindricalZAxis>(u) = st::get<CartesianZAxis>(v0);

		return std::move(u);
	}

};
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS>,
		coordinate_system::Cylindrical<IPhiAxis> >::CylindricalRAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS>,
		coordinate_system::Cylindrical<IPhiAxis> >::CylindricalZAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS>,
		coordinate_system::Cylindrical<IPhiAxis> >::CylindricalPhiAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS>,
		coordinate_system::Cylindrical<IPhiAxis> >::CartesianXAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS>,
		coordinate_system::Cylindrical<IPhiAxis> >::CartesianYAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<coordinate_system::Cartesian<3, ICARTESIAN_ZAXIS>,
		coordinate_system::Cylindrical<IPhiAxis> >::CartesianZAxis;
template<typename > struct mertic;

template<size_t IPhiAxis>
struct mertic<coordinate_system::template Cylindrical<IPhiAxis> >
{

	typedef gt::point_t<coordinate_system::Cylindrical<IPhiAxis> > point_t;
	typedef gt::vector_t<coordinate_system::Cylindrical<IPhiAxis> > vector_t;
	typedef gt::covector_t<coordinate_system::Cylindrical<IPhiAxis> > covector_t;

	typedef nTuple<Real, 3> delta_t;

	static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
	static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
	static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;

private:
	static constexpr Real dl_(integer_sequence<size_t, CylindricalRAxis>,
			point_t const & x0, delta_t const & delta)
	{
		return st::get<CylindricalRAxis>(delta);
	}

	static constexpr Real dl_(integer_sequence<size_t, CylindricalZAxis>,
			point_t const & x0, delta_t const & delta)
	{
		return st::get<CylindricalZAxis>(delta);
	}

	static constexpr Real dl_(integer_sequence<size_t, CylindricalPhiAxis>,
			point_t const & x0, delta_t const & delta)
	{

		return st::get<CylindricalRAxis>(x0)
				* st::get<CylindricalPhiAxis>(delta);

	}
public:
	template<size_t DI>
	static constexpr Real dl(point_t const & x0, delta_t const & delta)
	{

		return dl_(integer_sequence<size_t, DI>(), x0, delta);
	}

	template<typename ...Others>
	static constexpr Real volume(size_t node_id, point_t const & x0,
			delta_t delta, Others && ...)
	{

		return (((node_id >> CylindricalRAxis) & 1UL)
				* (dl<CylindricalRAxis>(x0, delta)
						* st::get<CylindricalRAxis>(delta) - 1.0) + 1.0)

				* (((node_id >> CylindricalZAxis) & 1UL)
						* (dl<CylindricalZAxis>(x0, delta)
								* st::get<CylindricalZAxis>(delta) - 1.0) + 1.0)

				* (((node_id >> CylindricalPhiAxis) & 1UL)
						* (dl<CylindricalPhiAxis>(x0, delta)
								* st::get<CylindricalPhiAxis>(delta) - 1.0)
						+ 1.0)

		;

	}

	template<typename ...Others>
	static constexpr Real dual_volume(size_t node_id, Others && ...others)
	{
		return volume(7UL & (~node_id), std::forward<Others>(others)...);
	}

};

}  // namespace geometry
namespace traits
{

template<size_t IPhiAxis>
struct description<geometry::coordinate_system::template Cylindrical<IPhiAxis> >
{
	static   std::string name()
	{
		return "Cylindrical<" + simpla::type_cast<std::string>(IPhiAxis) + ">";
	}
};

}  // namespace traits
}  // namespace simpla

#endif /* CORE_GEOMETRY_CS_CYLINDRICAL_H_ */