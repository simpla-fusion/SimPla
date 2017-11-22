//
// Created by salmon on 17-7-9.
//

#ifndef SIMPLA_TOKAMAK_H
#define SIMPLA_TOKAMAK_H

#include <simpla/SIMPLA_config.h>

#include <memory>
#include <string>
#include <utility>

#include <simpla/algebra/nTuple.h>
#include <simpla/data/Configurable.h>
#include <simpla/data/DataEntry.h>
#include <simpla/data/Serializable.h>
#include <simpla/geometry/Axis.h>
#include <simpla/geometry/gPolygon2D.h>
#include <simpla/geometry/GeoEntity.h>
#include <simpla/utilities/SPDefines.h>
#include <functional>
namespace simpla {

/**
 *  Define Tokamak geometry, parser GEqdsk
 */
class Tokamak : public data::Serializable, public data::Configurable {
    SP_SERIALIZABLE_HEAD(data::Serializable, Tokamak);
    SP_ENABLE_NEW;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;

   protected:
    explicit Tokamak(std::string const &url = "");

   public:
    ~Tokamak() override;

    void ReadGFile(std::string const &);
    void WriteGFile(std::string const &) const;

    void ReadProfile(std::string const &fname);

    geometry::Axis GetAxis() const;
    std::shared_ptr<const geometry::gPolygon2D> Limiter() const;
    std::shared_ptr<const geometry::gPolygon2D> Boundary() const;

    std::function<vector_type(point_type const &)> B0() const;
    std::function<Real(point_type const &)> profile(std::string const &k) const;

    //    std::shared_ptr<geometry::gPolygon> const &boundary() const;
    //    std::shared_ptr<geometry::gPolygon> const &limiter() const;
    //     bool in_boundary(point_type const &x) const { return boundary()->IsInside(x[RAxis],
    //    x[ZAxis]); }
    //     bool in_limiter(point_type const &x) const { return limiter()->IsInside(x[RAxis],
    //    x[ZAxis]); }
    Real psi(Real R, Real Z) const;
    Real psi(point_type const &x) const { return psi(x[RAxis], x[ZAxis]); }
    nTuple<Real, 2> grad_psi(Real R, Real Z) const;
    Real profile(std::string const &name, Real p_psi) const;
    Real profile(std::string const &name, Real R, Real Z) const { return profile(name, psi(R, Z)); }
    Real profile(std::string const &name, point_type const &x) const { return profile(name, psi(x[RAxis], x[ZAxis])); }
    std::function<Real(point_type const &x)> GetAttribute(std::string const &) const;

    /**
      *
      * @param R
      * @param Z
      * @return magnetic field on cylindrical coordinates \f$\left(R,Z,\phi\right)\f$
      */
    vector_type B(Real R, Real Z) const {
        auto gradPsi = grad_psi(R, Z);

        vector_type res;
        res[RAxis] = gradPsi[1] / R;
        res[ZAxis] = -gradPsi[0] / R;
        res[PhiAxis] = profile("fpol", psi(R, Z));

        return std::move(res);
    }

    vector_type B(point_type const &x) const {
        Real R = x[RAxis];
        Real Z = x[ZAxis];
        Real Phi = x[PhiAxis];

        auto gradPsi = grad_psi(R, Z);
        Real v_r = gradPsi[1] / R;
        Real v_z = -gradPsi[0] / R;
        Real v_phi = profile("fpol", psi(R, Z));

        vector_type res;

        res[RAxis] = v_r;
        res[ZAxis] = v_z;
        res[PhiAxis] = v_phi;

        //        res[CartesianXAxis] = v_r * std::cos(Phi) - v_phi * std::sin(Phi);
        //        res[CartesianYAxis] = v_r * std::sin(Phi) + v_phi * std::cos(Phi);
        //        res[CartesianZAxis] = v_z;

        return std::move(res);
    }

    Real JT(Real R, Real Z) const { return R * profile("pprim", psi(R, Z)) + profile("ffprim", psi(R, Z)) / R; }

    //    inline Real JT(point_type const &x) const
    //    {
    //        Real R = x[RAxis];
    //        Real Z = x[ZAxis];
    //        Real Phi = x[PhiAxis];
    //
    //        return R * profile("pprim", psi(R, Z)) + GetAttribute("ffprim", psi(R, Z)) / R;
    //    }

    /**
     *  diff_scheme the contour at \f$\Psi_{j}\in\left[0,1\right]\f$
     *  \cite  Jardin:2010:CMP:1855040
     * @param psi_j \f$\Psi_j\in\left[0,1\right]\f$
     * @param M  \f$\theta_{i}=i2\pi/N\f$,
     * @param res points coordinats
     *
     * @param ToPhiAxis \f$\in\left(0,1,2\right)\f$,ToPhiAxis the \f$\phi\f$ coordinates component  of result
     * coordinats,
     * @param resolution
     * @return   if success return true, else return false
     *
     * \todo need improve!!  only valid for internal flux surface \f$\psi \le 1.0\f$; need x-point support
     */
    //    bool flux_surface(Real psi_j, size_t M, point_type *res, Real resoluton = 0.001);

    /**
     *
     *
     * @param surface 	flux surface constituted by  poings on RZ coordinats
     * @param res 		flux surface constituted by points on flux coordiantes
     * @param h 		\f$h\left(\psi,\theta\right)=h\left(R,Z\right)\f$ , input
     * @param PhiAxis
     * @return  		if success return true, else return false
     * \note Ref. page. 133 in \cite  Jardin:2010:CMP:1855040
     *  \f$h\left(\psi,\theta\right)\f$  | \f$\theta\f$
     *	------------- | -------------
     *	\f$R/\left|\nabla \psi\right| \f$  | constant arc length
     *	\f$R^2\f$  | straight field lines
     *	\f$R\f$ | constant area
     *   1  | constant volume
     *
     */
    //    bool map_to_flux_coordiantes(std::vector<point_type> const &surface,
    //                                 std::vector<point_type> *res,
    //                                 std::function<Real(Real, Real)> const &h, size_t PhiAxis = 2);

   private:
    static constexpr int PhiAxis = 2;
    static constexpr int RAxis = (PhiAxis + 1) % 3;
    static constexpr int ZAxis = (PhiAxis + 2) % 3;
    static constexpr int CartesianZAxis = 2;
    static constexpr int CartesianXAxis = (CartesianZAxis + 1) % 3;
    static constexpr int CartesianYAxis = (CartesianZAxis + 2) % 3;
};

}  // namespace simpla
#endif  // SIMPLA_TOKAMAK_H
