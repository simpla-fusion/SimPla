/*
 * read_geqdsk.h
 *
 *  created on: 2013-11-29
 *      Author: salmon
 */

#ifndef GEQDSK_H_
#define GEQDSK_H_

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"
#include "../numeric/interpolation.h"

namespace simpla
{

/**
 * @ingroup Model
 * @{
 *   \defgroup GEqdsk  GEqdsk
 * @}
 *
 * @ingroup GEqdsk
 * \brief GEqdsk file paser
 *  default using cylindrical coordinates \f$R,Z,\phi\f$
 * \note http://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
 */
class GEqdsk
{

public:

	typedef nTuple<Real, 3> coordinate_type;

	typedef Interpolation<LinearInterpolation, Real, Real> inter_type;

	typedef MultiDimesionInterpolation<BiLinearInterpolation, Real> inter2d_type;

private:
	static constexpr size_t PhiAxis = 2;
	static constexpr size_t RAxis = (PhiAxis + 1) % 3;
	static constexpr size_t ZAxis = (PhiAxis + 2) % 3;

	nTuple<size_t, 3> m_dims_ = { 1, 1, 1 };
	coordinate_type m_rzmin_;
	coordinate_type m_rzmax_;

	bool is_valid_ = false;
	std::string m_desc_;
//	size_t nw;//!< Number of horizontal R grid  points
//	size_t nh;//!< Number of vertical Z grid points
	Real m_rdim_; //!< Horizontal dimension in meter of computational box
	Real m_zdim_; //!< Vertical dimension in meter of computational box
	Real m_rleft_; //!< Minimum R in meter of rectangular computational box
	Real m_zmid_; //!< Z of center of computational box in meter
	Real m_rmaxis_ = 1.0; //!< R of magnetic axis in meter
	Real zmaxis = 1.0; //!< Z of magnetic axis in meter
//	Real simag;//!< Poloidal flux at magnetic axis in Weber / rad
//	Real sibry;//!< Poloidal flux at the plasma boundary in Weber / rad
	Real m_rcenter_ = 0.5; //!< R in meter of  vacuum toroidal magnetic field BCENTR
	Real m_bcenter_ = 0.5; //!< Vacuum toroidal magnetic field in Tesla at RCENTR
	Real m_current_ = 1.0; //!< Plasma current in Ampere

//	coordinate_type rzmin_;
//	coordinate_type rzmax_;

//	inter_type fpol_; //!< Poloidal current function in m-T $F=RB_T$ on flux grid
//	inter_type pres_;//!< Plasma pressure in $nt/m^2$ on uniform flux grid
//	inter_type ffprim_;//!< $FF^\prime(\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid
//	inter_type pprim_;//!< $P^\prime(\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid

	inter2d_type psirz_; //!< Poloidal flux in Webber/rad on the rectangular grid points

//	inter_type qpsi_;//!< q values on uniform flux grid from axis to boundary

	std::vector<coordinate_type> m_rzbbb_; //!< R,Z of boundary points in meter
	std::vector<coordinate_type> m_rzlim_; //!< R,Z of surrounding limiter contour in meter

	std::map<std::string, inter_type> m_profile_;

public:
	GEqdsk()
	{

	}
	GEqdsk(std::string const &fname)
	{
		load(fname);
	}
	template<typename TDict>
	GEqdsk(TDict const &dict)
	{
		load(dict["File"].template as<std::string>());
	}

	~GEqdsk()
	{
	}

//	std::string save(std::string const & path) const;

	void load(std::string const &fname);
//
//	void Write(std::string const &fname);

	void load_profile(std::string const &fname);

	inline Real profile(std::string const & name,
			coordinate_type const & x) const
	{
		return profile(name, psi(x[RAxis], x[ZAxis]));
	}
	inline Real profile(std::string const & name, Real R, Real Z) const
	{
		return profile(name, psi(R, Z));
	}

	inline Real profile(std::string const & name, Real p_psi) const
	{
		return m_profile_.at(name)(p_psi);
	}

	std::string const &description() const
	{
		return m_desc_;
	}

	nTuple<size_t, 3> const & dimensins() const
	{
		return m_dims_;
	}
	std::pair<coordinate_type, coordinate_type> extents() const
	{
		return std::make_pair(m_rzmin_, m_rzmax_);
	}
	bool is_valid() const
	{
		return is_valid_;
	}

	std::ostream & print(std::ostream & os);

	inline std::vector<coordinate_type> const & boundary() const
	{
		return m_rzbbb_;
	}
	inline std::vector<coordinate_type> const & limiter() const
	{
		return m_rzlim_;
	}

	inline Real psi(Real R, Real Z) const
	{
		return psirz_.calculate(R, Z);
	}

	inline Real psi(coordinate_type const&x) const
	{
		return psirz_.calculate(x[RAxis], x[ZAxis]);
	}

	/**
	 *
	 * @param R
	 * @param Z
	 * @return magenetic field on cylindrical coordiantes \f$\left(R,Z,\phi\right)\f$
	 */
	inline Vec3 B(Real R, Real Z) const
	{
		auto gradPsi = psirz_.grad(R, Z);

		Vec3 res;
		res[RAxis] = gradPsi[1] / R;
		res[ZAxis] = -gradPsi[0] / R;
		res[PhiAxis] = profile("fpol", R, Z);

		return std::move(res);

	}

	inline auto B(coordinate_type const&x) const
	DECL_RET_TYPE(B(x[RAxis], x[ZAxis] ))
	;

	inline Real JT(Real R, Real Z) const
	{
		return R * profile("pprim", R, Z) + profile("ffprim", R, Z) / R;
	}

	inline auto JT(coordinate_type const&x) const
	DECL_RET_TYPE(JT(x[RAxis], x[ZAxis]))
	;

	bool check_profile(std::string const & name) const
	{
		return (name == "psi") || (name == "JT") || (name == "B")
				|| (m_profile_.find(name) != m_profile_.end());
	}

	coordinate_type map_cylindrical_to_flux(
			coordinate_type const & psi_theta_phi, size_t VecZAxis = 2) const;

	coordinate_type map_flux_from_cylindrical(coordinate_type const & x,
			size_t VecZAxis = 2) const;
	/**
	 *  caculate the contour at \f$\Psi_{j}\in\left[0,1\right]\f$
	 *  \cite  Jardin:2010:CMP:1855040
	 * @param psi_j \f$\Psi_j\in\left[0,1\right]\f$
	 * @param M  \f$\theta_{i}=i2\pi/N\f$,
	 * @param res points coordinats
	 *
	 * @param ToPhiAxis \f$\in\left(0,1,2\right)\f$,ToPhiAxis the \f$\phi\f$ coordinates component  of result coordinats,
	 * @param resoluton
	 * @return   if success return true, else return false
	 *
	 * \todo need improve!!  only valid for internal flux surface \f$\psi \le 1.0\f$; need x-point support
	 */
	bool flux_surface(Real psi_j, size_t M, coordinate_type*res,
			size_t ToPhiAxis = 2, Real resoluton = 0.001);

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
	bool map_to_flux_coordiantes(std::vector<coordinate_type> const&surface,
			std::vector<coordinate_type> *res,
			std::function<Real(Real, Real)> const & h, size_t PhiAxis = 2);

}
;
//template<typename TF>
//void GEqdsk::get_profile_(std::integral_constant<bool, true>,
//		std::string const & name, TF* f) const
//{
//	if (name == "B")
//	{
//
//		f->pull_back(*this, [this](coordinate_type const & x)
//		{	return this->B(x);});
//
//	}
//	else
//	{
//		WARNING << "Geqdsk:  Object '" << name << "'[scalar]  does not exist!";
//	}
//
//}

//template<typename TF>
//void GEqdsk::get_profile_(std::integral_constant<bool, false>,
//		std::string const & name, TF* f) const
//{
//
//	if (name == "psi")
//	{
//
//		f->pull_back(*this, [this](coordinate_type const & x)
//		{	return this->psi(x);});
//
//	}
//	else if (name == "JT")
//	{
//
//		f->pull_back(*this, [this](coordinate_type const & x)
//		{	return this->JT(x);});
//	}
//	else if (check_profile(name))
//	{
//
//		f->pull_back(*this, [this,name](coordinate_type const & x)
//		{	return this->profile(name,x);});
//
//	}
//	else
//	{
//		WARNING << "Geqdsk:  Object '" << name << "'[scalar]  does not exist!";
//	}
//
//}
//
//std::string XDMFWrite(GEqdsk const & self, std::string const &fname,
//		size_t flag);

}
// namespace simpla

#endif /* GEQDSK_H_ */
