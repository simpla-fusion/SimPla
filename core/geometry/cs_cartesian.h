/**
 * @file cs_cartesian.h
 *
 *  Created on: 2015-6-14
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CS_CARTESIAN_H_
#define CORE_GEOMETRY_CS_CARTESIAN_H_

#include "../gtl/macro.h"
#include "../gtl/type_cast.h"
#include "coordinate_system.h"

namespace simpla
{


namespace st = simpla::traits;
namespace gt = simpla::traits;
//
//template<typename, typename> struct map;
//
//
//template<size_t ZAXIS0, size_t ZAXIS1>
//struct map<coordinate_system::Cartesian<3, ZAXIS0>,
//        coordinate_system::Cartesian<3, ZAXIS1> >
//{
//
//    static constexpr size_t CartesianZAxis0 = (ZAXIS0) % 3;
//    static constexpr size_t CartesianYAxis0 = (CartesianZAxis0 + 2) % 3;
//    static constexpr size_t CartesianXAxis0 = (CartesianZAxis0 + 1) % 3;
//    typedef gt::point_t<coordinate_system::Cartesian<3, ZAXIS0> > point_t0;
//    typedef gt::vector_t<coordinate_system::Cartesian<3, ZAXIS0> > vector_t0;
//    typedef gt::covector_t<coordinate_system::Cartesian<3, ZAXIS0> > covector_t0;
//
//    static constexpr size_t CartesianZAxis1 = (ZAXIS1) % 3;
//    static constexpr size_t CartesianYAxis1 = (CartesianZAxis1 + 2) % 3;
//    static constexpr size_t CartesianXAxis1 = (CartesianZAxis1 + 1) % 3;
//    typedef gt::point_t<coordinate_system::Cartesian<3, ZAXIS1> > point_t1;
//    typedef gt::vector_t<coordinate_system::Cartesian<3, ZAXIS1> > vector_t1;
//    typedef gt::covector_t<coordinate_system::Cartesian<3, ZAXIS1> > covector_t1;
//
//    static point_t1 eval(point_t0 const &x)
//    {
//        /**
//         *  @note
//         * coordinates transforam
//         *
//         *  \f{eqnarray*}{
//         *		x & = & r\cos\phi\\
//             *		y & = & r\sin\phi\\
//             *		z & = & Z
//         *  \f}
//         *
//         */
//        point_t1 y;
//
//        st::get<CartesianXAxis1>(y) = st::get<CartesianXAxis0>(x);
//
//        st::get<CartesianYAxis1>(y) = st::get<CartesianYAxis0>(x);
//
//        st::get<CartesianZAxis1>(y) = st::get<CartesianZAxis0>(x);
//
//        return std::move(y);
//    }
//
//    point_t1 operator()(point_t0 const &x) const
//    {
//        return eval(x);
//    }
//
//    template<typename TFun>
//    auto pull_back(point_t0 const &x0, TFun const &fun)
//    DECL_RET_TYPE ((fun(this->operator()(x0))))
//
//    template<typename TRect>
//    TRect pull_back(point_t0 const &x0,
//                    std::function<TRect(point_t0 const &)> const &fun)
//    {
//        return fun(this->operator()(x0));
//    }
//
//    /**
//     *
//     *   push_forward vector from Cylindrical  to Cartesian
//     * @param R  \f$ v=v_{r}\partial_{r}+v_{Z}\partial_{Z}+v_{\theta}/r\partial_{\theta} \f$
//     * @param CartesianZAxis
//     * @return  \f$ \left(x,y,z\right),u=u_{x}\partial_{x}+u_{y}\partial_{y}+u_{z}\partial_{z} \f$
//     *
//     */
//    vector_t1 push_forward(point_t0 const &x0, vector_t0 const &v)
//    {
//
//        vector_t1 u;
//
//        st::get<CartesianXAxis1>(u) = st::get<CartesianXAxis0>(v);
//        st::get<CartesianYAxis1>(u) = st::get<CartesianYAxis0>(v);
//        st::get<CartesianZAxis1>(u) = st::get<CartesianZAxis0>(v);
//
//        return std::move(u);
//    }
//
//};

template<typename...> struct Metric;

template<size_t ICARTESIAN_ZAXIS>
struct Metric<coordinate_system::template Cartesian<3, ICARTESIAN_ZAXIS> >
{

    static constexpr size_t CartesianZAxis = (ICARTESIAN_ZAXIS) % 3;
    static constexpr size_t CartesianYAxis = (CartesianZAxis + 2) % 3;
    static constexpr size_t CartesianXAxis = (CartesianZAxis + 1) % 3;
    typedef gt::point_t<coordinate_system::template Cartesian<3, ICARTESIAN_ZAXIS> > point_t;
    typedef gt::vector_t<coordinate_system::template Cartesian<3, ICARTESIAN_ZAXIS> > vector_t;
    typedef gt::covector_t<coordinate_system::template Cartesian<3, ICARTESIAN_ZAXIS> > covector_t;

    typedef nTuple<Real, 3> delta_t;

    static constexpr Real metric_length(point_t const &p0, point_t const &p1)
    {
        return std::sqrt(dot(p1 - p0, p1 - p0));
    }

    static constexpr Real metric_area(point_t const &p0, point_t const &p1, point_t const &p2, point_t const &p3)
    {
        // FIXME assume  box is orthogonal
        return std::sqrt(dot(cross(p1 - p0, p3 - p0), cross(p1 - p0, p3 - p0)));
    }

    static constexpr Real metric_volume(point_t const &p0, point_t const &p1, point_t const &p2, point_t const &p3,
                                        point_t const &p4, point_t const &p5, point_t const &p6, point_t const &p7)
    {
        // FIXME assume  box is orthogonal
        return dot(p4 - p0, cross(p1 - p0, p2 - p0));
    }


    template<typename T0, typename T1, typename ...Others>
    static constexpr auto inner_product(T0 const &v0, T1 const &v1, Others &&... others)
    DECL_RET_TYPE((v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]))

};

namespace traits
{

template<size_t NDIMS, size_t ICARTESIAN_ZAXIS>
struct type_id<coordinate_system::Cartesian < NDIMS, ICARTESIAN_ZAXIS> >
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
