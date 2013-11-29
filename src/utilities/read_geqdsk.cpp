/*
 * read_geqdsk.cpp
 *
 *  Created on: 2013年11月29日
 *      Author: salmon
 */

#include "read_geqdsk.h"
#include "pertty_ostream.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
//#include <memory>
#include <string>
#include <vector>

namespace simpla
{

void GEqdsk::Read()
{
	inFileStream_.get(desc, 48);

	int idum;

	inFileStream_ >> std::setw(4) >> idum >> nw >> nh;

	double xdum;

	inFileStream_ >> std::setw(16)

	>> rdim >> zdim >> rcentr >> rleft >> zmid

	>> rmaxis >> zmaxis >> simag >> sibry >> bcentr

	>> current >> simag >> xdum >> rmaxis >> xdum

	>> zmaxis >> xdum >> sibry >> xdum >> xdum;

	fpol.resize(nw);
	pres.resize(nw);
	ffprim.resize(nw);
	pprim.resize(nw);
	qpsi.resize(nw);
	psirz.resize(nw * nh);

	inFileStream_

	>> std::setw(16) >> fpol

	>> std::setw(16) >> pres

	>> std::setw(16) >> ffprim

	>> std::setw(16) >> pprim

	>> std::setw(16) >> psirz

	>> std::setw(16) >> qpsi;

	inFileStream_ >> std::setw(5) >> nbbbs >> limitr;

	rzbbb.resize(nbbbs * 2);
	rzlim.resize(limitr * 2);

	inFileStream_ >> std::setw(16) >> rzbbb >> rzlim;
}
std::ostream & GEqdsk::Print(std::ostream & os)
{
	std::cout << "--" << desc << std::endl;

	std::cout << "nw" << "\t= " << nw
			<< "\t--  Number of horizontal R grid  points" << std::endl;

	std::cout << "nh" << "\t= " << nh << "\t-- Number of vertical Z grid points"
			<< std::endl;

	std::cout << "rdim" << "\t= " << rdim
			<< "\t-- Horizontal dimension in meter of computational box                   "
			<< std::endl;

	std::cout << "zdim" << "\t= " << zdim
			<< "\t-- Vertical dimension in meter of computational box                   "
			<< std::endl;

	std::cout << "rcentr" << "\t= " << rcentr
			<< "\t--                                                                    "
			<< std::endl;

	std::cout << "rleft" << "\t= " << rleft
			<< "\t-- Minimum R in meter of rectangular computational box                "
			<< std::endl;

	std::cout << "zmid" << "\t= " << zmid
			<< "\t-- Z of center of computational box in meter                          "
			<< std::endl;

	std::cout << "rmaxis" << "\t= " << rmaxis
			<< "\t-- R of magnetic axis in meter                                        "
			<< std::endl;

	std::cout << "rmaxis" << "\t= " << zmaxis
			<< "\t-- Z of magnetic axis in meter                                        "
			<< std::endl;

	std::cout << "simag" << "\t= " << simag
			<< "\t-- poloidal flus ax magnetic axis in Weber / rad                      "
			<< std::endl;

	std::cout << "sibry" << "\t= " << sibry
			<< "\t-- Poloidal flux at the plasma boundary in Weber / rad                "
			<< std::endl;

	std::cout << "rcentr" << "\t= " << rcentr
			<< "\t-- R in meter of  vacuum toroidal magnetic field BCENTR               "
			<< std::endl;

	std::cout << "bcentr" << "\t= " << bcentr
			<< "\t-- Vacuum toroidal magnetic field in Tesla at RCENTR                  "
			<< std::endl;

	std::cout << "current" << "\t= " << current
			<< "\t-- Plasma current in Ampere                                          "
			<< std::endl;

	std::cout << "fpol" << "\t= "
			<< "\t-- Poloidal current function in m-T<< $F=RB_T$ on flux grid           "
			<< std::endl << fpol << std::endl;

	std::cout << "pres" << "\t= "
			<< "\t-- Plasma pressure in $nt/m^2$ on uniform flux grid                   "
			<< std::endl << pres << std::endl;

	std::cout << "ffprim" << "\t= "
			<< "\t-- $FF^\\prime(\\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid     "
			<< std::endl << ffprim << std::endl;

	std::cout << "pprim" << "\t= "
			<< "\t-- $P^\\prime(\\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid    "
			<< std::endl << pprim << std::endl;

	std::cout << "psizr"
			<< "\t-- Poloidal flus in Webber/rad on the rectangular grid points         "
			<< std::endl << psirz << std::endl;

	std::cout << "qpsi" << "\t= "
			<< "\t-- q values on uniform flux grid from axis to boundary                "
			<< std::endl << qpsi << std::endl;

	std::cout << "nbbbs" << "\t= " << nbbbs
			<< "\t-- Number of boundary points                                          "
			<< std::endl;

	std::cout << "limitr" << "\t= " << limitr
			<< "\t-- Number of limiter points                                           "
			<< std::endl;

	std::cout << "rzbbbs" << "\t= "
			<< "\t-- R of boundary points in meter                                      "
			<< std::endl << rzbbb << std::endl;

	std::cout << "rzlim" << "\t= "
			<< "\t-- R of surrounding limiter contour in meter                          "
			<< std::endl << rzlim << std::endl;

	return os;
}
}  // namespace simpla
using namespace simpla;
int main(int argc, char ** argv)
{
	GEqdsk geqdsk(argv[1]);

	geqdsk.Read();
	geqdsk.Print(std::cout);
}
