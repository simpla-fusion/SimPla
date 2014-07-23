/*
 * read_geqdsk.h
 *
 *  created on: 2013-11-29
 *      Author: salmon
 */

#ifndef GEQDSK_H_
#define GEQDSK_H_

#include <iostream>
#include <string>
#include <vector>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../numeric/interpolation.h"
#include "../physics/constants.h"
#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cylindrical.h"

namespace simpla
{

/**
 * \ingroup Model
 * @{
 *   \defgroup GEqdsk  GEqdsk
 * @}
 *
 * \ingroup GEqdsk
 * \brief GEqdsk file paser
 *  default using cylindrical coordinates \f$R,Z,\phi\f$
 * \note http://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
 */
class GEqdsk: public CylindricalGeometry<UniformArray, 2>
{

public:
	typedef CylindricalGeometry<UniformArray, 2> geometry_type;

	typedef Interpolation<LinearInterpolation, Real, Real> inter_type;

	typedef MultiDimesionInterpolation<BiLinearInterpolation, Real> inter2d_type;

private:
	bool is_ready_ = false;
	std::string desc;
//	size_t nw;//!< Number of horizontal R grid  points
//	size_t nh;//!< Number of vertical Z grid points
	Real rdim; //!< Horizontal dimension in meter of computational box
	Real zdim; //!< Vertical dimension in meter of computational box
	Real rleft; //!< Minimum R in meter of rectangular computational box
	Real zmid; //!< Z of center of computational box in meter
	Real rmaxis = 1.0; //!< R of magnetic axis in meter
	Real zmaxis = 1.0; //!< Z of magnetic axis in meter
//	Real simag;//!< Poloidal flux at magnetic axis in Weber / rad
//	Real sibry;//!< Poloidal flux at the plasma boundary in Weber / rad
	Real rcenter = 0.5; //!< R in meter of  vacuum toroidal magnetic field BCENTR
	Real bcenter = 0.5; //!< Vacuum toroidal magnetic field in Tesla at RCENTR
	Real current = 1.0; //!< Plasma current in Ampere

//	coordinates_type rzmin_;
//	coordinates_type rzmax_;

//	inter_type fpol_; //!< Poloidal current function in m-T $F=RB_T$ on flux grid
//	inter_type pres_;//!< Plasma pressure in $nt/m^2$ on uniform flux grid
//	inter_type ffprim_;//!< $FF^\prime(\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid
//	inter_type pprim_;//!< $P^\prime(\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid

	inter2d_type psirz_; //!< Poloidal flux in Webber/rad on the rectangular grid points

//	inter_type qpsi_;//!< q values on uniform flux grid from axis to boundary

	std::vector<coordinates_type> rzbbb_; //!< R,Z of boundary points in meter
	std::vector<coordinates_type> rzlim_; //!< R,Z of surrounding limiter contour in meter

	std::map<std::string, inter_type> profile_;

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

	std::string save(std::string const & path) const;

	void load(std::string const &fname);

	void Write(std::string const &fname);

	void load_profile(std::string const &fname);

	inline Real Profile(std::string const & name, coordinates_type const & x) const
	{
		return Profile(name, psi(x[RAxis], x[ZAxis]));
	}
	inline Real Profile(std::string const & name, Real R, Real Z) const
	{
		return Profile(name, psi(R, Z));
	}

	inline Real Profile(std::string const & name, Real p_psi) const
	{
		return profile_.at(name)(p_psi);
	}

	std::string const &Description() const
	{
		return desc;
	}

	bool is_ready() const
	{
		return is_ready_;

	}

	std::ostream & print(std::ostream & os);

	inline std::vector<coordinates_type> const & Boundary() const
	{
		return rzbbb_;
	}
	inline std::vector<coordinates_type> const & Limiter() const
	{
		return rzlim_;
	}

	inline Real psi(Real R, Real Z) const
	{
		return psirz_.eval(R, Z);
	}

	inline Real psi(coordinates_type const&x) const
	{
		return psirz_.eval(x[RAxis], x[ZAxis]);
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
		res[PhiAxis] = Profile("fpol", R, Z);

		return std::move(res);

	}

	inline auto B(coordinates_type const&x) const
	DECL_RET_TYPE(B(x[RAxis], x[ZAxis] ))
	;

	inline Real JT(Real R, Real Z) const
	{
		return R * Profile("pprim", R, Z) + Profile("ffprim", R, Z) / R;
	}

	inline auto JT(coordinates_type const&x) const
	DECL_RET_TYPE(JT(x[RAxis], x[ZAxis]))
	;

	bool CheckProfile(std::string const & name) const
	{
		return (name == "psi") || (name == "JT") || (name == "B") || (profile_.find(name) != profile_.end());
	}

	template<typename TModel>
	void SetUpMaterial(TModel *model, unsigned int toridal_model_number = 0,
	        unsigned int DestPhiAxis = CARTESIAN_ZAXIS) const;

	template<typename TF>
	void GetProfile(std::string const & name, TF* f) const
	{
		GetProfile_(std::integral_constant<bool, is_nTuple<typename TF::field_value_type>::value>(), name, f);
		updateGhosts(f);
	}

	coordinates_type MapCylindricalToFlux(coordinates_type const & psi_theta_phi, unsigned int VecZAxis = 2) const;

	coordinates_type MapFluxFromCylindrical(coordinates_type const & x, unsigned int VecZAxis = 2) const;
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
	bool FluxSurface(Real psi_j, size_t M, coordinates_type*res, unsigned int ToPhiAxis = 2, Real resoluton = 0.001);

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
	bool MapToFluxCoordiantes(std::vector<coordinates_type> const&surface, std::vector<coordinates_type> *res,
	        std::function<Real(Real, Real)> const & h, unsigned int PhiAxis = 2);
private:

	template<typename TF>
	void GetProfile_(std::integral_constant<bool, true>, std::string const & name, TF* f) const;
	template<typename TF>
	void GetProfile_(std::integral_constant<bool, false>, std::string const & name, TF* f) const;
}
;
template<typename TModel>
void GEqdsk::SetUpMaterial(TModel *model, unsigned int toridal_model_number, unsigned int DestPhiAxis) const
{
	model->Set(model->SelectByPolylines(VERTEX, Limiter()), model->RegisterMaterial("Vacuum"));

	model->Set(model->SelectByPolylines(VERTEX, Boundary()), model->RegisterMaterial("Plasma"));

}
template<typename TF>
void GEqdsk::GetProfile_(std::integral_constant<bool, true>, std::string const & name, TF* f) const
{
	if (name == "B")
	{

		f->pull_back(*this, [this](coordinates_type const & x)
		{	return this->B(x);});

	}
	else
	{
		WARNING << "Geqdsk:  Object '" << name << "'[scalar]  does not exist!";
	}

}

template<typename TF>
void GEqdsk::GetProfile_(std::integral_constant<bool, false>, std::string const & name, TF* f) const
{

	if (name == "psi")
	{

		f->pull_back(*this, [this](coordinates_type const & x)
		{	return this->psi(x);});

	}
	else if (name == "JT")
	{

		f->pull_back(*this, [this](coordinates_type const & x)
		{	return this->JT(x);});
	}
	else if (CheckProfile(name))
	{

		f->pull_back(*this, [this,name](coordinates_type const & x)
		{	return this->Profile(name,x);});

	}
	else
	{
		WARNING << "Geqdsk:  Object '" << name << "'[scalar]  does not exist!";
	}

}

std::string XDMFWrite(GEqdsk const & self, std::string const &fname, unsigned int flag);

}
// namespace simpla

#endif /* GEQDSK_H_ */
