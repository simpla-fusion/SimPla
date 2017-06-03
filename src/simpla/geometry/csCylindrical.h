/**
 * @file csCylindrical.h
 *
 *  Created on: 2015-6-13
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CS_CYLINDRICAL_H_
#define CORE_GEOMETRY_CS_CYLINDRICAL_H_

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/nTuple.h>

#include "Chart.h"
#include "CoordinateSystem.h"
#include "simpla/utilities/macro.h"
#include "simpla/utilities/type_traits.h"
namespace simpla {
namespace geometry {

/** @ingroup   coordinate_system
 ** @{
 *  Metric of  Cylindrical topology_coordinate system
 */

/**
 *  RZPhi
 */
struct Cylindrical : public Chart {
    SP_OBJECT_HEAD(Cylindrical, Chart)
   public:
    typedef Real scalar_type;

    SP_DEFAULT_CONSTRUCT(Cylindrical);
    DECLARE_REGISTER_NAME("Cylindrical")

    static constexpr int PhiAxis = 2;
    static constexpr int RAxis = (PhiAxis + 1) % 3;
    static constexpr int ZAxis = (PhiAxis + 2) % 3;

    template <typename... Args>
    explicit Cylindrical(Args &&... args) : Chart(std::forward<Args>(args)...) {}
    ~Cylindrical() override = default;

   public:
    /**
     *  from local coordinates to global Cartesian coordinates
     */
    point_type map(point_type const &uvw) const override {
        return Chart::map(
            point_type{uvw[RAxis] * std::cos(uvw[PhiAxis]), uvw[RAxis] * std::sin(uvw[PhiAxis]), uvw[ZAxis]});
    }

    /**
     *  from  global Cartesian coordinates to local coordinates
     * @param uvw
     * @return
     */
    point_type inv_map(point_type const &xyz) const override {
        point_type r = Chart::inv_map(xyz);
        point_type uvw;
        uvw[PhiAxis] = std::atan2(r[1], r[0]);
        uvw[RAxis] = std::hypot(r[0], r[1]);
        uvw[ZAxis] = r[2];
        return uvw;
    }

    Real length(point_type const &p0, point_type const &p1) const override {
        Real r0 = p0[RAxis];
        Real z0 = p0[ZAxis];
        Real phi0 = p0[PhiAxis];

        Real dr1 = p1[RAxis] - r0;
        Real dz1 = p1[ZAxis] - z0;
        Real dphi1 = p1[PhiAxis] - phi0;

        Real a = std::sqrt(power2(dr1) + power2(dz1) + power2(r0 * dphi1));

        return a /*1st*/ + power2(dphi1) * dr1 * r0 / (2 * a) /*2nd*/;
    }

    Real simplex_area(point_type const &p0, point_type const &p1, point_type const &p2) const override {
        Real r0 = p0[RAxis];
        Real z0 = p0[ZAxis];
        Real phi0 = p0[PhiAxis];

        Real dr1 = p1[RAxis] - r0;
        Real dz1 = p1[ZAxis] - z0;
        Real dphi1 = p1[PhiAxis] - phi0;

        Real dr2 = p2[RAxis] - r0;
        Real dz2 = p2[ZAxis] - z0;
        Real dphi2 = p2[PhiAxis] - phi0;

        Real A = std::sqrt(power2(r0) * (power2(dphi1 * dr2 - dphi2 * dr1) + power2(dphi1 * dz2 - dphi2 * dz1)) +
                           power2(dr1 * dz2 - dr2 * dz1));

        return

            // 2nd
            0.5 * A

            // 3rd
            +
            r0 * (dr1 + dr2) * (power2(dphi1 * dr2 - dphi2 * dr1) + power2(dphi1 * dz2 - dphi2 * dz1)) /
                (6 * A) /* 3rd */

            //                // 4th
            //                + power2(dr1 + dr2) * power2(dr1 * dz2 - dr2 * dz1) *
            //                  (power2(dphi1 * dr2 - dphi2 * dr1) + power2(dphi1 * dz2 - dphi2 * dz1)) / (24 *
            //                  power3(A))
            //
            //                // 5th
            //                - r0 * (dr1 + dr2) * (power2(dr1) + power2(dr2)) * power2(dr1 * dz2 - dr2 * dz1) *
            //                  power2(power2(dphi1 * dr2 - dphi2 * dr1) + power2(dphi1 * dz2 - dphi2 * dz1)) /
            //                  (40 * power3(A) * power2(A))
            //
            //            //

            ;
    }

    Real simplex_volume(point_type const &p0, point_type const &p1, point_type const &p2,
                        point_type const &p3) const override {
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

        return (dr1 + dr2 + dr3 + 4 * r0) * (dphi1 * dr2 * dz3 - dphi1 * dr3 * dz2 - dphi2 * dr1 * dz3 +
                                             dphi2 * dr3 * dz1 + dphi3 * dr1 * dz2 - dphi3 * dr2 * dz1) /
               24.0;
    }

    Real box_area(point_type const &p0, point_type const &p1) const override {
        Real r0 = min(p0[RAxis], p1[RAxis]);
        Real phi0 = min(p0[PhiAxis], p1[PhiAxis]);
        Real z0 = min(p0[RAxis], p1[RAxis]);

        Real r1 = max(p0[RAxis], p1[RAxis]);
        Real phi1 = max(p0[PhiAxis], p1[PhiAxis]);
        Real z1 = max(p0[ZAxis], p1[ZAxis]);

        if (std::abs(r1 - r0) < EPSILON) {
            return r0 * (phi1 - phi0) * (z1 - z0);
        } else if (std::abs(z1 - z0) < EPSILON) {
            return 0.5 * (power2(r1 - r0) + 2 * r0 * (r1 - r0)) * (phi1 - phi0);
        } else if (std::abs(phi1 - phi0) < EPSILON) {
            return (r1 - r0) * (z1 - z0);

        } else {
            //            THROW_EXCEPTION("Undefined result");
            return std::numeric_limits<Real>::quiet_NaN();
        }
    }

    Real box_volume(point_type const &p0, point_type const &p1) const override {
        Real r0 = min(p0[RAxis], p1[RAxis]);
        Real phi0 = min(p0[PhiAxis], p1[PhiAxis]);
        Real z0 = min(p0[RAxis], p1[RAxis]);

        Real r1 = max(p0[RAxis], p1[RAxis]);
        Real phi1 = max(p0[PhiAxis], p1[PhiAxis]);
        Real z1 = max(p0[ZAxis], p1[ZAxis]);

        return 0.5 * ((r1 - r0) * (r1 - r0) + 2 * r0 * (r1 - r0)) * (phi1 - phi0) * (z1 - z0);
    }

    Real inner_product(point_type const &uvw, vector_type const &v0, vector_type const &v1) const override {
        return 0;
        //        return std::abs(
        //            (v0[RAxis] * v1[RAxis] + v0[ZAxis] * v1[ZAxis] + v0[PhiAxis] * v1[PhiAxis] * r[RAxis] *
        //            r[RAxis]));
    }
};

}  // namespace geometry
}  // namespace simpla

//
// template<typename, typename> struct map;
//
// template<size_t IPhiAxis, size_t I_CARTESIAN_ZAXIS>
// struct map<Cylindrical<IPhiAxis>, Cartesian<3, I_CARTESIAN_ZAXIS> >
//{
//
//    typedef gt::point_type<Cylindrical<IPhiAxis> > point_t0;
//    typedef gt::vector_type<Cylindrical<IPhiAxis> > vector_t0;
//    typedef gt::covector_type<Cylindrical<IPhiAxis> > covector_t0;
//
//    static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
//    static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
//    static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;
//
//    typedef gt::point_type<Cartesian<3, CARTESIAN_XAXIS> >
//            point_t1;
//    typedef gt::vector_type<Cartesian<3, CARTESIAN_XAXIS> >
//            vector_t1;
//    typedef gt::covector_type<Cartesian<3, CARTESIAN_XAXIS> >
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
//        st::get<CartesianXAxis>(y) = st::Pop<CylindricalRAxis>(x)
//                                     * std::cos(st::Pop<CylindricalPhiAxis>(x));
//
//        st::get<CartesianYAxis>(y) = st::Pop<CylindricalRAxis>(x)
//                                     * std::sin(st::Pop<CylindricalPhiAxis>(x));
//
//        st::get<CartesianZAxis>(y) = st::Pop<CylindricalZAxis>(x);
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
//    AUTO_RETURN((fun(map(x0))))
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
// * @_fdtd_param R  \f$ v=v_{r}\partial_{r}+v_{Z}\partial_{Z}+v_{\theta}/r\partial_{\theta} \f$
// * @_fdtd_param CartesianZAxis
// * @return  \f$ \left(x,y,z\right),u=u_{x}\partial_{x}+u_{y}\partial_{y}+u_{z}\partial_{z} \f$
// *
// */
//    vector_t1 push_forward(point_t0 const &x0, vector_t0 const &v0)
//    {
//
//        Real cos_phi = std::cos(st::Pop<CylindricalPhiAxis>(x0));
//        Real sin_phi = std::cos(st::Pop<CylindricalPhiAxis>(x0));
//        Real r = st::Pop<CylindricalRAxis>(x0);
//
//        Real vr = st::Pop<CylindricalRAxis>(v0);
//        Real vphi = st::Pop<CylindricalPhiAxis>(v0);
//
//        vector_t1 u;
//
//        st::Pop<CartesianXAxis>(u) = vr * cos_phi - vphi * r * sin_phi;
//        st::Pop<CartesianYAxis>(u) = vr * sin_phi + vphi * r * cos_phi;
//        st::Pop<CartesianZAxis>(u) = st::get<CylindricalZAxis>(v0);
//
//        return std::Move(u);
//    }
//
//};
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CylindricalRAxis;
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CylindricalZAxis;
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CylindricalPhiAxis;
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CartesianXAxis;
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CartesianYAxis;
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cylindrical<IPhiAxis>, Cartesian<3, ICARTESIAN_ZAXIS>>::CartesianZAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// struct map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis> >
//{
//
//    typedef gt::point_type<Cylindrical<IPhiAxis>> point_t1;
//    typedef gt::vector_type<Cylindrical<IPhiAxis>> vector_t1;
//    typedef gt::covector_type<Cylindrical<IPhiAxis>> covector_t1;
//
//    static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
//    static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
//    static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;
//
//    typedef gt::point_type<Cartesian<3, CARTESIAN_XAXIS>> point_t0;
//    typedef gt::vector_type<Cartesian<3, CARTESIAN_XAXIS>> vector_t0;
//    typedef gt::covector_type<Cartesian<3, CARTESIAN_XAXIS>> covector_t0;
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
//        st::get<CylindricalZAxis>(y) = st::Pop<CartesianYAxis>(x);
//        st::Pop<CylindricalRAxis>(y) = std::sqrt(
//                st::get<CartesianXAxis>(x) * st::Pop<CartesianXAxis>(x)
//                + st::Pop<CartesianZAxis>(x)
//                  * st::Pop<CartesianZAxis>(x));
//        st::Pop<CylindricalPhiAxis>(y) = std::atan2(st::get<CartesianZAxis>(x),
//                                                    st::Pop<CartesianXAxis>(x));
//
//        return std::Move(y);
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
//    AUTO_RETURN((fun(map(x0))))
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
// * @_fdtd_param Z  \f$ \left(x,y,z\right),u=u_{x}\partial_{x}+u_{y}\partial_{y}+u_{z}\partial_{z} \f$
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
//        Real cos_phi = std::cos(st::Pop<CylindricalPhiAxis>(x0));
//        Real sin_phi = std::cos(st::Pop<CylindricalPhiAxis>(x0));
//
//        Real r = st::Pop<CylindricalRAxis>(x0);
//
//        vector_t1 u;
//
//        Real vx = st::Pop<CartesianXAxis>(v0);
//        Real vy = st::Pop<CartesianYAxis>(v0);
//
//        st::Pop<CylindricalPhiAxis>(u) = (-vx * sin_phi + vy * cos_phi) / r;
//
//        st::Pop<CylindricalRAxis>(u) = vx * cos_phi + vy * sin_phi;
//
//        st::Pop<CylindricalZAxis>(u) = st::get<CartesianZAxis>(v0);
//
//        return std::Move(u);
//    }
//
//};
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CylindricalRAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CylindricalZAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CylindricalPhiAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CartesianXAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CartesianYAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<Cartesian<3, ICARTESIAN_ZAXIS>, Cylindrical<IPhiAxis>>::CartesianZAxis;

#endif /* CORE_GEOMETRY_CS_CYLINDRICAL_H_ */
