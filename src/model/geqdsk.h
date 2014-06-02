/*
 * read_geqdsk.h
 *
 *  Created on: 2013年11月29日
 *      Author: salmon
 */

#ifndef GEQDSK_H_
#define GEQDSK_H_

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../numeric/interpolation.h"
#include "../io/data_stream.h"
namespace simpla
{

/**
 * @ref http://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
 */
class GEqdsk
{

public:

	typedef Real value_type;
	typedef Interpolation<LinearInterpolation, value_type, Real> inter_type;
	typedef MultiDimesionInterpolation<BiLinearInterpolation, value_type> inter2d_type;
	enum
	{
		NDIMS = 2
	};
private:
	std::string desc;
//	size_t nw; // Number of horizontal R grid  points
//	size_t nh; // Number of vertical Z grid points
//	Real rdim; // Horizontal dimension in meter of computational box
//	Real zdim; // Vertical dimension in meter of computational box
//	Real rleft; // Minimum R in meter of rectangular computational box
//	Real zmid; // Z of center of computational box in meter
	Real rmaxis; // R of magnetic axis in meter
	Real zmaxis; // Z of magnetic axis in meter
//	Real simag; // Poloidal flux at magnetic axis in Weber / rad
//	Real sibry; // Poloidal flux at the plasma boundary in Weber / rad
	Real rcentr; // R in meter of  vacuum toroidal magnetic field BCENTR
	Real bcentr; // Vacuum toroidal magnetic field in Tesla at RCENTR
	Real current; // Plasma current in Ampere

	nTuple<NDIMS, size_t> dims_ =
	{ 0, 0 };
	nTuple<NDIMS, Real> rzmin_ =
	{ 0, 0 };
	nTuple<NDIMS, Real> rzmax_ =
	{ 0, 0 };

//	inter_type fpol_; // Poloidal current function in m-T $F=RB_T$ on flux grid
//	inter_type pres_; // Plasma pressure in $nt/m^2$ on uniform flux grid
//	inter_type ffprim_; // $FF^\prime(\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid
//	inter_type pprim_; // $P^\prime(\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid

	inter2d_type psirz_; // Poloidal flux in Webber/rad on the rectangular grid points

//	inter_type qpsi_; // q values on uniform flux grid from axis to boundary

	std::vector<nTuple<NDIMS, Real> > rzbbb_; // R,Z of boundary points in meter
	std::vector<nTuple<NDIMS, Real> > rzlim_; // R,Z of surrounding limiter contour in meter

	std::map<std::string, inter_type> profile_;

public:

	GEqdsk(std::string const &fname = "")
	{
		Read(fname);
	}

	~GEqdsk()
	{
	}

	void Load(std::string const & fname)
	{
		Read(fname);
	}
	void Save(std::ostream & os = std::cout) const;

	void Read(std::string const &fname);

	void Write(std::string const &fname);

	void ReadProfile(std::string const &fname);

	inline value_type Profile(std::string const & name, Real x, Real y) const
	{
		return profile_.at(name)(psi(x, y));
	}

	inline value_type Profile(std::string const & name, Real p) const
	{
		return profile_.at(name)(p);
	}

	std::string const &Description() const
	{
		return desc;
	}
	nTuple<NDIMS, Real> const & GetMin() const
	{
		return rzmin_;
	}

	nTuple<NDIMS, Real> const &GetMax() const
	{
		return rzmax_;
	}

	nTuple<NDIMS, size_t> const &GetDimension() const
	{
		return dims_;
	}

	std::ostream & Print(std::ostream & os);

	inline std::vector<nTuple<NDIMS, Real> > const & Boundary() const
	{
		return rzbbb_;
	}
	inline std::vector<nTuple<NDIMS, Real> > const & Limiter() const
	{
		return rzlim_;
	}

	inline value_type psi(Real x, Real y) const
	{
		return psirz_.eval(x, y);
	}

	inline nTuple<3, Real> B(Real x, Real y) const
	{
		auto gradPsi = psirz_.diff(x, y);

		return nTuple<3, Real>(
		{

		gradPsi[1] / x,

		-gradPsi[0] / x,

		Profile("fpol", x, y) / x });

	}

	inline Real JT(Real x, Real y) const
	{
		return x * Profile("pprim", x, y) + Profile("ffprim", x, y) / x;
	}
};
std::string XDMFWrite(GEqdsk const & self, std::string const &fname, int flag);

}
// namespace simpla

#endif /* GEQDSK_H_ */
