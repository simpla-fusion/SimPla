/**
 * @file cs_cylindrical.h
 *
 *  Created on: 2015-6-13
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CS_CYLINDRICAL_H_
#define CORE_GEOMETRY_CS_CYLINDRICAL_H_

#include "../gtl/macro.h"
#include "../gtl/type_traits.h"
#include "coordinate_system.h"

namespace simpla
{


namespace st = simpla::traits;
namespace gt = simpla::traits;
using namespace coordinate_system;

template<typename, typename> struct map;

template<size_t IPhiAxis, size_t I_CARTESIAN_ZAXIS>
struct map<Cylindrical<IPhiAxis>, Cartesian<3, I_CARTESIAN_ZAXIS> >
{

    typedef gt::point_t<Cylindrical<IPhiAxis> > point_t0;
    typedef gt::vector_t<Cylindrical<IPhiAxis> > vector_t0;
    typedef gt::covector_t<Cylindrical<IPhiAxis> > covector_t0;

    static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
    static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
    static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;

    typedef gt::point_t<Cartesian<3, CARTESIAN_XAXIS> >
            point_t1;
    typedef gt::vector_t<Cartesian<3, CARTESIAN_XAXIS> >
            vector_t1;
    typedef gt::covector_t<Cartesian<3, CARTESIAN_XAXIS> >
            covector_t1;

    static constexpr size_t CartesianZAxis = (I_CARTESIAN_ZAXIS) % 3;
    static constexpr size_t CartesianYAxis = (CartesianZAxis + 2) % 3;
    static constexpr size_t CartesianXAxis = (CartesianZAxis + 1) % 3;

    static point_t1 eval(point_t0 const &x)
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

    point_t1 operator()(point_t0 const &x) const
    {
        return eval(x);
    }

    template<typename TFun>
    auto pull_back(point_t0 const &x0, TFun const &fun)
    DECL_RET_TYPE((fun(map(x0))))

    template<typename TRect>
    TRect pull_back(point_t0 const &x0, std::function<TRect(point_t0 const &)> const &fun)
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
    vector_t1 push_forward(point_t0 const &x0, vector_t0 const &v0)
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
constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CylindricalRAxis;
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CylindricalZAxis;
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CylindricalPhiAxis;
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CartesianXAxis;
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CartesianYAxis;
template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CartesianZAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
struct map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis> >
{

    typedef gt::point_t<Cylindrical<IPhiAxis>> point_t1;
    typedef gt::vector_t<Cylindrical<IPhiAxis>> vector_t1;
    typedef gt::covector_t<Cylindrical<IPhiAxis>> covector_t1;

    static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
    static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
    static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;

    typedef gt::point_t<Cartesian<3, CARTESIAN_XAXIS>> point_t0;
    typedef gt::vector_t<Cartesian<3, CARTESIAN_XAXIS>> vector_t0;
    typedef gt::covector_t<Cartesian<3, CARTESIAN_XAXIS>> covector_t0;

    static constexpr size_t CartesianZAxis = (ICARTESIAN_ZAXIS) % 3;
    static constexpr size_t CartesianYAxis = (CartesianZAxis + 2) % 3;
    static constexpr size_t CartesianXAxis = (CartesianZAxis + 1) % 3;

    static point_t1 eval(point_t0 const &x)
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

    point_t1 operator()(point_t0 const &x) const
    {
        return eval(x);
    }

    template<typename TFun>
    auto pull_back(point_t0 const &x0, TFun const &fun)
    DECL_RET_TYPE((fun(map(x0))))

    template<typename TRect>
    TRect pull_back(point_t0 const &x0,
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
    vector_t1 push_forward(point_t0 const &x0, vector_t0 const &v0)
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
constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CylindricalRAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CylindricalZAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CylindricalPhiAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CartesianXAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CartesianYAxis;

template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CartesianZAxis;


template<typename...> struct Metric;

template<size_t IPhiAxis>
struct Metric<Cylindrical<IPhiAxis> >
{
public:
    typedef gt::point_t<Cylindrical<IPhiAxis>> point_t;
    typedef gt::vector_t<Cylindrical<IPhiAxis>> vector_t;
    typedef gt::covector_t<Cylindrical<IPhiAxis>> covector_t;

    typedef nTuple<Real, 3> delta_t;

    static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
    static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
    static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;


    static constexpr Real simplex_length(point_t const &p0, point_t const &p1)
    {
        return std::sqrt(dot(p1 - p0, p1 - p0));
    }

    static constexpr Real simplex_area(point_t const &p0, point_t const &p1, point_t const &p2)
    {
        return (std::sqrt(dot(cross(p1 - p0, p2 - p0), cross(p1 - p0, p2 - p0)))) * 0.5;
    }


    static constexpr Real simplex_volume(point_t const &p0, point_t const &p1, point_t const &p2, point_t const &p3)
    {
        return dot(p3 - p0, cross(p1 - p0, p2 - p0)) / 6.0;
    }


    template<typename T0, typename T1, typename TX, typename ...Others>
    static constexpr Real inner_product(T0 const &v0, T1 const &v1, TX const &x, Others &&... others)
    {
        return ((v0[CylindricalRAxis] * v1[CylindricalRAxis] + v0[CylindricalZAxis] * v1[CylindricalZAxis] +
                 v0[CylindricalPhiAxis] * v1[CylindricalPhiAxis] * x[CylindricalRAxis] * x[CylindricalRAxis]));
    };


};

namespace traits
{

template<size_t IPhiAxis>
struct type_id<Cylindrical<IPhiAxis> >
{
    static std::string name()
    {
        return "Cylindrical<" + simpla::type_cast<std::string>(IPhiAxis) + ">";
    }
};

}  // namespace traits
}  // namespace simpla

#endif /* CORE_GEOMETRY_CS_CYLINDRICAL_H_ */
