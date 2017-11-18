/**
 * @file csCylindrical.h
 *
 *  Created on: 2015-6-13
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CS_CYLINDRICAL_H_
#define CORE_GEOMETRY_CS_CYLINDRICAL_H_

#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/nTuple.ext.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/utilities/SPDefines.h"
#include "simpla/utilities/macro.h"
#include "simpla/utilities/type_traits.h"

#include "Chart.h"

namespace simpla {
namespace geometry {

/** @ingroup   coordinate_system
 ** @{
 *  Metric of  Cylindrical topology_coordinate system
 */

/**
 *  RZPhi
 */
struct csCylindrical : public Chart {
    SP_SERIALIZABLE_HEAD(Chart, csCylindrical)
   protected:
    template <typename... Args>
    explicit csCylindrical(Args &&... args) : base_type(std::forward<Args>(args)...) {}
    csCylindrical();
    csCylindrical(csCylindrical const &);

   public:
    ~csCylindrical() override;
    template <typename... Args>
    static std::shared_ptr<this_type> New(Args &&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }
    typedef Real scalar_type;

    static constexpr int PhiAxis = 2;
    static constexpr int RAxis = (PhiAxis + 1) % 3;
    static constexpr int ZAxis = (PhiAxis + 2) % 3;

    std::shared_ptr<Edge> GetCoordinateEdge(point_type const &x0, int normal, Real u) const override;
    std::shared_ptr<Face> GetCoordinateFace(point_type const &x0, int normal, Real u, Real v) const override;
    std::shared_ptr<Solid> GetCoordinateBox(point_type const &o, Real u, Real v, Real w) const override;

   public:
    /**
     *  from local coordinates to global Cartesian coordinates
     */
    point_type map(point_type const &uvw) const override {
        return point_type{uvw[RAxis] * std::cos(uvw[PhiAxis]), uvw[RAxis] * std::sin(uvw[PhiAxis]), uvw[ZAxis]};
    }

    /**
     *  from  global Cartesian coordinates to local coordinates
     * @param uvw
     * @return
     */
    point_type inv_map(point_type const &r) const override {
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

    Real area(point_type const &p0, point_type const &p1, point_type const &p2) const override {
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

    Real volume(point_type const &p0, point_type const &p1, point_type const &p2, point_type const &p3) const override {
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

    Real length(point_type const &p0, point_type const &p1, int normal) const override {
        Real r0 = p0[RAxis];
        Real phi0 = p0[PhiAxis];
        Real z0 = p0[ZAxis];

        Real r1 = p1[RAxis];
        Real phi1 = p1[PhiAxis];
        Real z1 = p1[ZAxis];

        Real res = 0;
        switch (normal) {
            case RAxis:
                res = r1 - r0;
                break;
            case ZAxis:
                res = z1 - z0;
                break;
            case PhiAxis:
            default:
                res = r0 * (phi1 - phi0);
                break;
        }
        return res;
    }

    Real area(point_type const &p0, point_type const &p1, int normal = 0) const override {
        Real r0 = p0[RAxis];
        Real phi0 = p0[PhiAxis];
        Real z0 = p0[ZAxis];

        Real r1 = p1[RAxis];
        Real phi1 = p1[PhiAxis];
        Real z1 = p1[ZAxis];

        Real res = 0;
        switch (normal) {
            case RAxis:
                res = r0 * (phi1 - phi0) * (z1 - z0);
                break;
            case ZAxis:
                res = 0.5 * ((r1 - r0) * (r1 - r0) + 2 * r0 * (r1 - r0)) * (phi1 - phi0);
                break;
            case PhiAxis:
            default:
                res = (r1 - r0) * (z1 - z0);
                break;
        }
        return res;

        //        if (std::abs(r1 - r0) < EPSILON) {
        //            return r0 * (phi1 - phi0) * (z1 - z0);
        //        } else if (std::abs(z1 - z0) < EPSILON) {
        //            return 0.5 * (power2(r1 - r0) + 2 * r0 * (r1 - r0)) * (phi1 - phi0);
        //        } else if (std::abs(phi1 - phi0) < EPSILON) {
        //            return (r1 - r0) * (z1 - z0);
        //
        //        } else {
        //            //            THROW_EXCEPTION("Undefined result");
        //            return std::numeric_limits<Real>::quiet_NaN();
        //        }
    }

    Real volume(point_type const &p0, point_type const &p1) const override {
        Real r0 = p0[RAxis];
        Real phi0 = p0[PhiAxis];
        Real z0 = p0[ZAxis];

        Real r1 = p1[RAxis];
        Real phi1 = p1[PhiAxis];
        Real z1 = p1[ZAxis];

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
// struct map<csCylindrical<IPhiAxis>, csCartesian<3, I_CARTESIAN_ZAXIS> >
//{
//
//    typedef gt::point_type<csCylindrical<IPhiAxis> > point_t0;
//    typedef gt::vector_type<csCylindrical<IPhiAxis> > vector_t0;
//    typedef gt::covector_type<csCylindrical<IPhiAxis> > covector_t0;
//
//    static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
//    static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
//    static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;
//
//    typedef gt::point_type<csCartesian<3, CARTESIAN_XAXIS> >
//            point_t1;
//    typedef gt::vector_type<csCartesian<3, CARTESIAN_XAXIS> >
//            vector_t1;
//    typedef gt::covector_type<csCartesian<3, CARTESIAN_XAXIS> >
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
//        st::get<CartesianXAxis>(y) = st::Serialize<CylindricalRAxis>(x)
//                                     * std::cos(st::Serialize<CylindricalPhiAxis>(x));
//
//        st::get<CartesianYAxis>(y) = st::Serialize<CylindricalRAxis>(x)
//                                     * std::sin(st::Serialize<CylindricalPhiAxis>(x));
//
//        st::get<CartesianZAxis>(y) = st::Serialize<CylindricalZAxis>(x);
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
// *   push_forward vector from csCylindrical  to csCartesian
// * @_fdtd_param R  \f$ v=v_{r}\partial_{r}+v_{Z}\partial_{Z}+v_{\theta}/r\partial_{\theta} \f$
// * @_fdtd_param CartesianZAxis
// * @return  \f$ \left(x,y,z\right),u=u_{x}\partial_{x}+u_{y}\partial_{y}+u_{z}\partial_{z} \f$
// *
// */
//    vector_t1 push_forward(point_t0 const &x0, vector_t0 const &v0)
//    {
//
//        Real cos_phi = std::cos(st::Serialize<CylindricalPhiAxis>(x0));
//        Real sin_phi = std::cos(st::Serialize<CylindricalPhiAxis>(x0));
//        Real r = st::Serialize<CylindricalRAxis>(x0);
//
//        Real vr = st::Serialize<CylindricalRAxis>(v0);
//        Real vphi = st::Serialize<CylindricalPhiAxis>(v0);
//
//        vector_t1 u;
//
//        st::Serialize<CartesianXAxis>(u) = vr * cos_phi - vphi * r * sin_phi;
//        st::Serialize<CartesianYAxis>(u) = vr * sin_phi + vphi * r * cos_phi;
//        st::Serialize<CartesianZAxis>(u) = st::get<CylindricalZAxis>(v0);
//
//        return std::Move(u);
//    }
//
//};
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCylindrical<IPhiAxis>, csCartesian<3, ICARTESIAN_ZAXIS>>::CylindricalRAxis;
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCylindrical<IPhiAxis>, csCartesian<3, ICARTESIAN_ZAXIS>>::CylindricalZAxis;
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCylindrical<IPhiAxis>, csCartesian<3, ICARTESIAN_ZAXIS>>::CylindricalPhiAxis;
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCylindrical<IPhiAxis>, csCartesian<3, ICARTESIAN_ZAXIS>>::CartesianXAxis;
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCylindrical<IPhiAxis>, csCartesian<3, ICARTESIAN_ZAXIS>>::CartesianYAxis;
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCylindrical<IPhiAxis>, csCartesian<3, ICARTESIAN_ZAXIS>>::CartesianZAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// struct map<csCartesian<3, ICARTESIAN_ZAXIS>, csCylindrical<IPhiAxis> >
//{
//
//    typedef gt::point_type<csCylindrical<IPhiAxis>> point_t1;
//    typedef gt::vector_type<csCylindrical<IPhiAxis>> vector_t1;
//    typedef gt::covector_type<csCylindrical<IPhiAxis>> covector_t1;
//
//    static constexpr size_t CylindricalPhiAxis = (IPhiAxis) % 3;
//    static constexpr size_t CylindricalRAxis = (CylindricalPhiAxis + 1) % 3;
//    static constexpr size_t CylindricalZAxis = (CylindricalPhiAxis + 2) % 3;
//
//    typedef gt::point_type<csCartesian<3, CARTESIAN_XAXIS>> point_t0;
//    typedef gt::vector_type<csCartesian<3, CARTESIAN_XAXIS>> vector_t0;
//    typedef gt::covector_type<csCartesian<3, CARTESIAN_XAXIS>> covector_t0;
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
//        st::get<CylindricalZAxis>(y) = st::Serialize<CartesianYAxis>(x);
//        st::Serialize<CylindricalRAxis>(y) = std::sqrt(
//                st::get<CartesianXAxis>(x) * st::Serialize<CartesianXAxis>(x)
//                + st::Serialize<CartesianZAxis>(x)
//                  * st::Serialize<CartesianZAxis>(x));
//        st::Serialize<CylindricalPhiAxis>(y) = std::atan2(st::get<CartesianZAxis>(x),
//                                                    st::Serialize<CartesianXAxis>(x));
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
// *   push_forward vector from csCartesian to csCylindrical
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
//        Real cos_phi = std::cos(st::Serialize<CylindricalPhiAxis>(x0));
//        Real sin_phi = std::cos(st::Serialize<CylindricalPhiAxis>(x0));
//
//        Real r = st::Serialize<CylindricalRAxis>(x0);
//
//        vector_t1 u;
//
//        Real vx = st::Serialize<CartesianXAxis>(v0);
//        Real vy = st::Serialize<CartesianYAxis>(v0);
//
//        st::Serialize<CylindricalPhiAxis>(u) = (-vx * sin_phi + vy * cos_phi) / r;
//
//        st::Serialize<CylindricalRAxis>(u) = vx * cos_phi + vy * sin_phi;
//
//        st::Serialize<CylindricalZAxis>(u) = st::get<CartesianZAxis>(v0);
//
//        return std::Move(u);
//    }
//
//};
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCartesian<3, ICARTESIAN_ZAXIS>, csCylindrical<IPhiAxis>>::CylindricalRAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCartesian<3, ICARTESIAN_ZAXIS>, csCylindrical<IPhiAxis>>::CylindricalZAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCartesian<3, ICARTESIAN_ZAXIS>, csCylindrical<IPhiAxis>>::CylindricalPhiAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCartesian<3, ICARTESIAN_ZAXIS>, csCylindrical<IPhiAxis>>::CartesianXAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCartesian<3, ICARTESIAN_ZAXIS>, csCylindrical<IPhiAxis>>::CartesianYAxis;
//
// template<size_t IPhiAxis, size_t ICARTESIAN_ZAXIS>
// constexpr size_t map<csCartesian<3, ICARTESIAN_ZAXIS>, csCylindrical<IPhiAxis>>::CartesianZAxis;

#endif /* CORE_GEOMETRY_CS_CYLINDRICAL_H_ */
