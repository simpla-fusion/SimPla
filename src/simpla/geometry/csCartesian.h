/**
 * @file csCartesian.h
 *
 *  Created on: 2015-6-14
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CS_CARTESIAN_H_
#define CORE_GEOMETRY_CS_CARTESIAN_H_

#include <simpla/utilities/nTuple.h>
#include "Chart.h"
#include "simpla/utilities/macro.h"
#include "simpla/utilities/type_cast.h"
namespace simpla {
namespace geometry {

/**
 * @ingroup  coordinate_system
 * @{
 *  Metric of  Cartesian topology_coordinate system
 */
struct Cartesian : public Chart {
    SP_OBJECT_HEAD(Cartesian, Chart)
   public:
    typedef Real scalar_type;

    SP_DEFAULT_CONSTRUCT(Cartesian);
    DECLARE_REGISTER_NAME("Cartesian")

    /**
     * metric only diff_scheme the volume of simplex
     *
     */

    point_type map(point_type const &p) const override { return p; }
    point_type inv_map(point_type const &p) const override { return p; }

    Real length(point_type const &p0, point_type const &p1) const override { return std::sqrt(dot(p1 - p0, p1 - p0)); }

    Real simplex_area(point_type const &p0, point_type const &p1, point_type const &p2) const override {
        return (std::sqrt(dot(cross(p1 - p0, p2 - p0), cross(p1 - p0, p2 - p0)))) * 0.5;
    }

    Real simplex_volume(point_type const &p0, point_type const &p1, point_type const &p2,
                        point_type const &p3) const override {
        return dot(p3 - p0, cross(p1 - p0, p2 - p1)) / 6.0;
    }
};
/** @}*/
}
}  // namespace simpla

//
// template<typename, typename> struct map;
//
//
// template<int ZAXIS0, int ZAXIS1>
// struct map<coordinate_system::Cartesian<3, ZAXIS0>,
//        coordinate_system::Cartesian<3, ZAXIS1> >
//{
//
//    static constexpr int CartesianZAxis0 = (ZAXIS0) % 3;
//    static constexpr int CartesianYAxis0 = (CartesianZAxis0 + 2) % 3;
//    static constexpr int CartesianXAxis0 = (CartesianZAxis0 + 1) % 3;
//    typedef gt::point_type<coordinate_system::Cartesian<3, ZAXIS0> > point_t0;
//    typedef gt::vector_type<coordinate_system::Cartesian<3, ZAXIS0> > vector_t0;
//    typedef gt::covector_type<coordinate_system::Cartesian<3, ZAXIS0> > covector_t0;
//
//    static constexpr int CartesianZAxis1 = (ZAXIS1) % 3;
//    static constexpr int CartesianYAxis1 = (CartesianZAxis1 + 2) % 3;
//    static constexpr int CartesianXAxis1 = (CartesianZAxis1 + 1) % 3;
//    typedef gt::point_type<coordinate_system::Cartesian<3, ZAXIS1> > point_t1;
//    typedef gt::vector_type<coordinate_system::Cartesian<3, ZAXIS1> > vector_t1;
//    typedef gt::covector_type<coordinate_system::Cartesian<3, ZAXIS1> > covector_t1;
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
//        st::PopPatch<CartesianXAxis1>(y) = st::get<CartesianXAxis0>(x);
//
//        st::get<CartesianYAxis1>(y) = st::PopPatch<CartesianYAxis0>(x);
//
//        st::get<CartesianZAxis1>(y) = st::PopPatch<CartesianZAxis0>(x);
//
//        return std::Move(y);
//    }
//
//    point_t1 operator()(point_t0 const &x) const
//    {
//        return eval(x);
//    }
//
//    template<typename TFun>
//    auto pull_back(point_t0 const &x0, TFun const &fun)
//    AUTO_RETURN ((fun(this->operator()(x0))))
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
//     * @_fdtd_param R  \f$ v=v_{r}\partial_{r}+v_{Z}\partial_{Z}+v_{\theta}/r\partial_{\theta} \f$
//     * @_fdtd_param CartesianZAxis
//     * @return  \f$ \left(x,y,z\right),u=u_{x}\partial_{x}+u_{y}\partial_{y}+u_{z}\partial_{z} \f$
//     *
//     */
//    vector_t1 push_forward(point_t0 const &x0, vector_t0 const &v)
//    {
//
//        vector_t1 u;
//
//        st::get<CartesianXAxis1>(u) = st::PopPatch<CartesianXAxis0>(v);
//        st::PopPatch<CartesianYAxis1>(u) = st::get<CartesianYAxis0>(v);
//        st::PopPatch<CartesianZAxis1>(u) = st::get<CartesianZAxis0>(v);
//
//        return std::Move(u);
//    }
//
//};

#endif /* CORE_GEOMETRY_CS_CARTESIAN_H_ */
