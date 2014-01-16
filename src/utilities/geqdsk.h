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
//#include <map>
//#include <memory>
#include <string>
//#include <utility>
#include <vector>

#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../numeric/interpolation.h"
//#include "../simpla_defs.h"
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
	char desc[50];
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

	nTuple<NDIMS, size_t> dims_;
	nTuple<NDIMS, Real> rzmin_;
	nTuple<NDIMS, Real> rzmax_;

	inter_type fpol_; // Poloidal current function in m-T $F=RB_T$ on flux grid
	inter_type pres_; // Plasma pressure in $nt/m^2$ on uniform flux grid
	inter_type ffprim_; // $FF^\prime(\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid
	inter_type pprim_; // $P^\prime(\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid

	inter2d_type psirz_; // Poloidal flux in Webber/rad on the rectangular grid points

	inter_type qpsi_; // q values on uniform flux grid from axis to boundary

	std::vector<nTuple<NDIMS, Real> > rzbbb_; // R,Z of boundary points in meter
	std::vector<nTuple<NDIMS, Real> > rzlim_; // R,Z of surrounding limiter contour in meter

public:
	GEqdsk()
	{
	}
	GEqdsk(std::string const &fname = "")
	{
		Read(fname);
	}

	~GEqdsk()
	{
	}

	enum
	{
		XDMF = 1, HDF5 = 2
	};

	void Read(std::string const &fname);
	void Write(std::string const &fname, int format = XDMF);

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

	template<typename ...Args>
	inline value_type psi(Args const &...x)
	{
		return psirz_(std::forward<Args const &>(x)...);
	}

#define VALUE_FUNCTION(_NAME_)                                     \
	template<typename ...Args>                                     \
	inline value_type _NAME_(Args const &...x)                       \
	{                                                              \
		return _NAME_##_(psi(std::forward<Args const &>(x)...));       \
	}

	VALUE_FUNCTION(fpol);
	VALUE_FUNCTION(pres);
	VALUE_FUNCTION(ffprim);
	VALUE_FUNCTION(pprim);
	VALUE_FUNCTION(qpsi);
#undef VALUE_FUNCTION

	template<typename TX>
	nTuple<3,Real> B(TX const & x)const
	{
		auto B_p= psirz_.diff(x[0],x[1]);

		nTuple<3,Real> res=
		{	B_p[0],B_p[1],fpol_(x[0])/x[0]};

		return std::move(res);
	}
};

}
// namespace simpla

#endif /* GEQDSK_H_ */
