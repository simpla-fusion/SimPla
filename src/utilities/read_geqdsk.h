/*
 * read_geqdsk.h
 *
 *  Created on: 2013年11月29日
 *      Author: salmon
 */

#ifndef READ_GEQDSK_H_
#define READ_GEQDSK_H_
#include <fstream>
#include <vector>
namespace simpla
{

/**
 * @ref http://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
 */
class GEqdsk
{
	std::ifstream inFileStream_;
public:

	char desc[50];
	int nw; // Number of horizontal R grid  points
	int nh; // Number of vertical Z grid points
	double rdim; // Horizontal dimension in meter of computational box
	double zdim; // Vertical dimension in meter of computational box
	double rleft; // Minimum R in meter of rectangular computational box
	double zmid; // Z of center of computational box in meter
	double rmaxis; // R of magnetic axis in meter
	double zmaxis; // Z of magnetic axis in meter
	double simag; // Poloidal flux at magnetic axis in Weber / rad
	double sibry; // Poloidal flux at the plasma boundary in Weber / rad
	double rcentr; // R in meter of  vacuum toroidal magnetic field BCENTR
	double bcentr; // Vacuum toroidal magnetic field in Tesla at RCENTR
	double current; // Plasma current in Ampere
	std::vector<double> fpol; // Poloidal current function in m-T $F=RB_T$ on flux grid
	std::vector<double> pres; // Plasma pressure in $nt/m^2$ on uniform flux grid
	std::vector<double> ffprim; // $FF^\prime(\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid
	std::vector<double> pprim; // $P^\prime(\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid
	std::vector<double> psirz; // Poloidal flux in Webber/rad on the rectangular grid points
	std::vector<double> qpsi; // q values on uniform flux grid from axis to boundary
	int nbbbs; // Number of boundary points
	int limitr; // Number of limiter points
	std::vector<double> rzbbb; // R,Z of boundary points in meter
	std::vector<double> rzlim; // R,Z of surrounding limiter contour in meter

	GEqdsk(std::string const &fname = "")
	{
		Open(fname);
	}

	~GEqdsk()
	{
	}
	void Open(std::string const &fname)
	{
		if (fname != "")
			inFileStream_.open(fname);
	}
	void Read();
	std::ostream & Print(std::ostream & os);
};

}  // namespace simpla

#endif /* READ_GEQDSK_H_ */
