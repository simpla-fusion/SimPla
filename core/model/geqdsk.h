/**
 *  @file geqdsk.h
 *
 *  created on: 2013-11-29
 *      Author: salmon
 */

#ifndef GEQDSK_H_
#define GEQDSK_H_

#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"
#include "../geometry/geo_object.h"

namespace simpla
{

/**
 * @ingroup Model
 * @{
 *   @defgroup GEqdsk  GEqdsk
 * @}
 *
 * @ingroup GEqdsk
 * \brief GEqdsk file paser
 *  default using cylindrical coordinates \f$R,Z,\phi\f$
 * \note http://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
 */
class GEqdsk
{


private:

    typedef GEqdsk this_type;

    typedef nTuple<Real, 3> point_type;

    static constexpr int PhiAxis = 2;
    static constexpr int RAxis = (PhiAxis + 1) % 3;
    static constexpr int ZAxis = (PhiAxis + 2) % 3;

public:

    GEqdsk();

    ~GEqdsk();

    void load(std::string const &fname);

    void load_profile(std::string const &fname);

    std::ostream &print(std::ostream &os);

    std::string const &description() const;

    geometry::Object const &boundary() const;

    geometry::Object const &limiter() const;

    Real psi(Real R, Real Z) const;

    Vec3 grad_psi(Real R, Real Z) const;

    Real profile(std::string const &name, Real p_psi) const;

    Real profile(std::string const &name, Real R, Real Z) const { return profile(name, psi(R, Z)); }

    point_type magnetic_axis() const;


    /**
     *
     * @param R
     * @param Z
     * @return magnetic field on cylindrical coordinates \f$\left(R,Z,\phi\right)\f$
     */
    inline Vec3 B(Real R, Real Z) const
    {
        auto gradPsi = grad_psi(R, Z);

        Vec3 res;
        res[RAxis] = gradPsi[1] / R;
        res[ZAxis] = -gradPsi[0] / R;
        res[PhiAxis] = profile("fpol", psi(R, Z));

        return std::move(res);

    }


    inline Real JT(Real R, Real Z) const
    {
        return R * profile("pprim", psi(R, Z)) + profile("ffprim", psi(R, Z)) / R;
    }

    /**
     *  diff_scheme the contour at \f$\Psi_{j}\in\left[0,1\right]\f$
     *  \cite  Jardin:2010:CMP:1855040
     * @param psi_j \f$\Psi_j\in\left[0,1\right]\f$
     * @param M  \f$\theta_{i}=i2\pi/N\f$,
     * @param res points coordinats
     *
     * @param ToPhiAxis \f$\in\left(0,1,2\right)\f$,ToPhiAxis the \f$\phi\f$ coordinates component  of result coordinats,
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
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};


}
// namespace simpla

#endif /* GEQDSK_H_ */
