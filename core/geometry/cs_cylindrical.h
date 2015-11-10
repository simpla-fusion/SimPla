/**
 * @file cs_cylindrical.h
 *
 *  Created on: 2015-6-13
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CS_CYLINDRICAL_H_
#define CORE_GEOMETRY_CS_CYLINDRICAL_H_

#include "../gtl/macro.h"
#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"
#include "coordinate_system.h"

namespace simpla
{


namespace st = simpla::traits;
namespace gt = simpla::traits;

template<typename...> struct Metric;

template<size_t IPhiAxis>
struct Metric<coordinate_system::Cylindrical<IPhiAxis> >
{
public:
    typedef coordinate_system::Cylindrical<IPhiAxis> cs;
    typedef gt::point_t<coordinate_system::Cylindrical<IPhiAxis>> point_t;
    typedef gt::vector_t<coordinate_system::Cylindrical<IPhiAxis>> vector_t;
    typedef gt::covector_t<coordinate_system::Cylindrical<IPhiAxis>> covector_t;

    typedef nTuple<Real, 3> delta_t;

    static constexpr size_t PhiAxis = cs::PhiAxis;
    static constexpr size_t RAxis = cs::RAxis;
    static constexpr size_t ZAxis = cs::ZAxis;


    static Real simplex_length(point_t const &p0, point_t const &p1)
    {
        Real r0 = p0[RAxis];
        Real z0 = p0[ZAxis];
        Real phi0 = p0[PhiAxis];

        Real dr1 = p1[RAxis] - r0;
        Real dz1 = p1[ZAxis] - z0;
        Real dphi1 = p1[PhiAxis] - phi0;


        Real a = std::sqrt(power2(dr1) + power2(dz1) + power2(r0 * dphi1));

        return a /*1st*/ + power2(dphi1) * dr1 * r0 / (2 * a) /*2nd*/
                ;
    }

    static Real simplex_area(point_t const &p0, point_t const &p1, point_t const &p2)
    {

        Real r0 = p0[RAxis];
        Real z0 = p0[ZAxis];
        Real phi0 = p0[PhiAxis];

        Real dr1 = p1[RAxis] - r0;
        Real dz1 = p1[ZAxis] - z0;
        Real dphi1 = p1[PhiAxis] - phi0;

        Real dr2 = p2[RAxis] - r0;
        Real dz2 = p2[ZAxis] - z0;
        Real dphi2 = p2[PhiAxis] - phi0;


        Real A = std::sqrt(
                power2(r0) * (power2(dphi1 * dr2 - dphi2 * dr1) + power2(dphi1 * dz2 - dphi2 * dz1)) +
                power2(dr1 * dz2 - dr2 * dz1));


        return

            // 2nd
                0.5 * A

                // 3rd
                + r0 * (dr1 + dr2) * (power2(dphi1 * dr2 - dphi2 * dr1) + power2(dphi1 * dz2 - dphi2 * dz1)) /
                  (6 * A) /* 3rd */

//                // 4th
//                + power2(dr1 + dr2) * power2(dr1 * dz2 - dr2 * dz1) *
//                  (power2(dphi1 * dr2 - dphi2 * dr1) + power2(dphi1 * dz2 - dphi2 * dz1)) / (24 * power3(A))
//
//                // 5th
//                - r0 * (dr1 + dr2) * (power2(dr1) + power2(dr2)) * power2(dr1 * dz2 - dr2 * dz1) *
//                  power2(power2(dphi1 * dr2 - dphi2 * dr1) + power2(dphi1 * dz2 - dphi2 * dz1)) /
//                  (40 * power3(A) * power2(A))
//
//            //

                ;


    }


    static Real simplex_volume(point_t const &p0, point_t const &p1, point_t const &p2, point_t const &p3)
    {

        Real r0 = p0[RAxis];
        Real phi0 = p0[PhiAxis];
        Real z0 = p0[ZAxis];

        Real dr1 = p1[RAxis] - r0;
        Real dphi1 = p1[PhiAxis] - phi0;
        Real dz1 = p1[ZAxis] - z0;

        Real dr2 = p2[RAxis] - r0;
        Real dphi2 = p2[PhiAxis] - phi0;
        Real dz2 = p2[ZAxis] - z0;

        Real dr3 = p3[RAxis] - r0;
        Real dphi3 = p3[PhiAxis] - phi0;
        Real dz3 = p3[ZAxis] - z0;

        return (dr1 + dr2 + dr3 + 4 * r0) *
               (dphi1 * dr2 * dz3 - dphi1 * dr3 * dz2 - dphi2 * dr1 * dz3 + dphi2 * dr3 * dz1 + dphi3 * dr1 * dz2 -
                dphi3 * dr2 * dz1) / 24.0;
    }

    static Real box_area(point_t const &p0, point_t const &p1)
    {

        Real r0 = min(p0[RAxis], p1[RAxis]);
        Real phi0 = min(p0[PhiAxis], p1[PhiAxis]);
        Real z0 = min(p0[RAxis], p1[RAxis]);

        Real r1 = max(p0[RAxis], p1[RAxis]);
        Real phi1 = max(p0[PhiAxis], p1[PhiAxis]);
        Real z1 = max(p0[ZAxis], p1[ZAxis]);

        if (std::abs(r1 - r0) < EPSILON)
        {
            return r0 * (phi1 - phi0) * (z1 - z0);

        }
        else if (std::abs(z1 - z0) < EPSILON)
        {
            return 0.5 * (power2(r1 - r0) + 2 * r0 * (r1 - r0)) * (phi1 - phi0);
        }
        else if (std::abs(phi1 - phi0) < EPSILON)
        {

            return (r1 - r0) * (z1 - z0);

        }
        else
        {
            ERROR("Undefined result");
            return 0;
        }
    }

    static Real box_volume(point_t const &p0, point_t const &p1)
    {

        Real r0 = min(p0[RAxis], p1[RAxis]);
        Real phi0 = min(p0[PhiAxis], p1[PhiAxis]);
        Real z0 = min(p0[RAxis], p1[RAxis]);

        Real r1 = max(p0[RAxis], p1[RAxis]);
        Real phi1 = max(p0[PhiAxis], p1[PhiAxis]);
        Real z1 = max(p0[ZAxis], p1[ZAxis]);


        return 0.5 * ((r1 - r0) * (r1 - r0) + 2 * r0 * (r1 - r0)) * (phi1 - phi0) * (z1 - z0);
    }

    template<typename T0, typename T1, typename TX, typename ...Others>
    static constexpr Real inner_product(T0 const &v0, T1 const &v1, TX const &r, Others &&... others)
    {
        return std::abs((v0[RAxis] * v1[RAxis] + v0[ZAxis] * v1[ZAxis] +
                         v0[PhiAxis] * v1[PhiAxis] * r[RAxis] * r[RAxis]));
    };


};

namespace traits
{

template<size_t IPhiAxis>
struct type_id<coordinate_system::Cylindrical<IPhiAxis> >
{
    static std::string name()
    {
        return "Cylindrical<" + simpla::type_cast<std::string>(IPhiAxis) + ">";
    }
};

}  // namespace traits
}  // namespace simpla


//
//template<typename, typename> struct map;
//
//template<size_t IPhiAxis, size_t I_CARTESIAN_ZAXIS>
//struct map<Cylindrical<IPhiAxis>, Cartesian<3, I_CARTESIAN_ZAXIS> >
//{
//
//    typedef gt::point_t<Cylindrical<IPhiAxis> > point_t0;
//    typedef gt::vector_t<Cylindrical<IPhiAxis> > vector_t0;
//    typedef gt::covector_t<Cylindrical<IPhiAxis> > covector_t0;
//
//    static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
//    static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
//    static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;
//
//    typedef gt::point_t<Cartesian<3, CARTESIAN_XAXIS> >
//            point_t1;
//    typedef gt::vector_t<Cartesian<3, CARTESIAN_XAXIS> >
//            vector_t1;
//    typedef gt::covector_t<Cartesian<3, CARTESIAN_XAXIS> >
//            covector_t1;
//
//    static constexpr size_t CartesianZAxis = (I_CARTESIAN_ZAXIS) % 3;
//    static constexpr size_t CartesianYAxis = (CartesianZAxis + 2) % 3;
//    static constexpr size_t CartesianXAxis = (CartesianZAxis + 1) % 3;
//
//    static point_t1 eval(point_t0 const &x)
//    {
///**
// *  @note
// * coordinates transforam
// *
// *  \f{eqnarray*}{
// *		x & = & r\cos\phi\\
//	 *		y & = & r\sin\phi\\
//	 *		z & = & Z
// *  \f}
// *
// */
//        point_t1 y;
//
//        st::get<CartesianXAxis>(y) = st::get<CylindricalRAxis>(x)
//                                     * std::cos(st::get<CylindricalPhiAxis>(x));
//
//        st::get<CartesianYAxis>(y) = st::get<CylindricalRAxis>(x)
//                                     * std::sin(st::get<CylindricalPhiAxis>(x));
//
//        st::get<CartesianZAxis>(y) = st::get<CylindricalZAxis>(x);
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
//    DECL_RET_TYPE((fun(map(x0))))
//
//    template<typename TRect>
//    TRect pull_back(point_t0 const &x0, std::function<TRect(point_t0 const &)> const &fun)
//    {
//        return fun(map(x0));
//    }
//
///**
// *
// *   push_forward vector from Cylindrical  to Cartesian
// * @param R  \f$ v=v_{r}\partial_{r}+v_{Z}\partial_{Z}+v_{\theta}/r\partial_{\theta} \f$
// * @param CartesianZAxis
// * @return  \f$ \left(x,y,z\right),u=u_{x}\partial_{x}+u_{y}\partial_{y}+u_{z}\partial_{z} \f$
// *
// */
//    vector_t1 push_forward(point_t0 const &x0, vector_t0 const &v0)
//    {
//
//        Real cos_phi = std::cos(st::get<CylindricalPhiAxis>(x0));
//        Real sin_phi = std::cos(st::get<CylindricalPhiAxis>(x0));
//        Real r = st::get<CylindricalRAxis>(x0);
//
//        Real vr = st::get<CylindricalRAxis>(v0);
//        Real vphi = st::get<CylindricalPhiAxis>(v0);
//
//        vector_t1 u;
//
//        st::get<CartesianXAxis>(u) = vr * cos_phi - vphi * r * sin_phi;
//        st::get<CartesianYAxis>(u) = vr * sin_phi + vphi * r * cos_phi;
//        st::get<CartesianZAxis>(u) = st::get<CylindricalZAxis>(v0);
//
//        return std::move(u);
//    }
//
//};
//
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CylindricalRAxis;
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CylindricalZAxis;
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CylindricalPhiAxis;
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CartesianXAxis;
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CartesianYAxis;
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CartesianZAxis;
//
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//struct map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis> >
//{
//
//    typedef gt::point_t<Cylindrical<IPhiAxis>> point_t1;
//    typedef gt::vector_t<Cylindrical<IPhiAxis>> vector_t1;
//    typedef gt::covector_t<Cylindrical<IPhiAxis>> covector_t1;
//
//    static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
//    static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
//    static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;
//
//    typedef gt::point_t<Cartesian<3, CARTESIAN_XAXIS>> point_t0;
//    typedef gt::vector_t<Cartesian<3, CARTESIAN_XAXIS>> vector_t0;
//    typedef gt::covector_t<Cartesian<3, CARTESIAN_XAXIS>> covector_t0;
//
//    static constexpr size_t CartesianZAxis = (ICARTESIAN_ZAXIS) % 3;
//    static constexpr size_t CartesianYAxis = (CartesianZAxis + 2) % 3;
//    static constexpr size_t CartesianXAxis = (CartesianZAxis + 1) % 3;
//
//    static point_t1 eval(point_t0 const &x)
//    {
//        point_t1 y;
//        /**
//         *  @note
//         *  coordinates transforam
//         *  \f{eqnarray*}{
//         *		r&=&\sqrt{x^{2}+y^{2}}\\
//         *		Z&=&z\\
//         *		\phi&=&\arg\left(y,x\right)
//         *  \f}
//         *
//         */
//        st::get<CylindricalZAxis>(y) = st::get<CartesianYAxis>(x);
//        st::get<CylindricalRAxis>(y) = std::sqrt(
//                st::get<CartesianXAxis>(x) * st::get<CartesianXAxis>(x)
//                + st::get<CartesianZAxis>(x)
//                  * st::get<CartesianZAxis>(x));
//        st::get<CylindricalPhiAxis>(y) = std::atan2(st::get<CartesianZAxis>(x),
//                                                    st::get<CartesianXAxis>(x));
//
//        return std::move(y);
//
//    }
//
//    point_t1 operator()(point_t0 const &x) const
//    {
//        return eval(x);
//    }
//
//    template<typename TFun>
//    auto pull_back(point_t0 const &x0, TFun const &fun)
//    DECL_RET_TYPE((fun(map(x0))))
//
//    template<typename TRect>
//    TRect pull_back(point_t0 const &x0,
//                    std::function<TRect(point_t0 const &)> const &fun)
//    {
//        return fun(map(x0));
//    }
//
///**
// *
// *   push_forward vector from Cartesian to Cylindrical
// *
// *
// * \verbatim
// *
// *     theta   y   r
// *          \  |  /
// *           \ | /
// *            \|/------x
// *          y  /
// *          | /
// *          |/)theta
// *          0------x
// *
// * \endverbatim
// *
// * @param Z  \f$ \left(x,y,z\right),u=u_{x}\partial_{x}+u_{y}\partial_{y}+u_{z}\partial_{z} \f$
// * @return  \f$ v=v_{r}\partial_{r}+v_{Z}\partial_{Z}+v_{\theta}/r\partial_{\theta} \f$
// *
// */
//    vector_t1 push_forward(point_t0 const &x0, vector_t0 const &v0)
//    {
//
//        point_t1 y;
//
//        y = map(x0);
//
//        Real cos_phi = std::cos(st::get<CylindricalPhiAxis>(x0));
//        Real sin_phi = std::cos(st::get<CylindricalPhiAxis>(x0));
//
//        Real r = st::get<CylindricalRAxis>(x0);
//
//        vector_t1 u;
//
//        Real vx = st::get<CartesianXAxis>(v0);
//        Real vy = st::get<CartesianYAxis>(v0);
//
//        st::get<CylindricalPhiAxis>(u) = (-vx * sin_phi + vy * cos_phi) / r;
//
//        st::get<CylindricalRAxis>(u) = vx * cos_phi + vy * sin_phi;
//
//        st::get<CylindricalZAxis>(u) = st::get<CartesianZAxis>(v0);
//
//        return std::move(u);
//    }
//
//};
//
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CylindricalRAxis;
//
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CylindricalZAxis;
//
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CylindricalPhiAxis;
//
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CartesianXAxis;
//
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CartesianYAxis;
//
//template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
//constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CartesianZAxis;


#endif /* CORE_GEOMETRY_CS_CYLINDRICAL_H_ */
