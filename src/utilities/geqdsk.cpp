/*
 * read_geqdsk.cpp
 *
 *  Created on: 2013年11月29日
 *      Author: salmon
 */

#include "geqdsk.h"

#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <utility>

#include "log.h"
#include "pretty_stream.h"

namespace simpla
{

void GEqdsk::Read(std::string const &fname)
{

	std::ifstream inFileStream_(fname);

	if (!inFileStream_.is_open())
	{
		ERROR << "File " << fname << " is not opend!";
		return;
	}

	LOGGER << "Load GFile : " << fname;

	int nw; //Number of horizontal R grid points
	int nh; //Number of vertical Z grid points
	double rdim; // Horizontal dimension in meter of computational box
	double zdim; // Vertical dimension in meter of computational box
	double rleft; // Minimum R in meter of rectangular computational box
	double zmid; // Z of center of computational box in meter
	double simag; // Poloidal flux at magnetic axis in Weber / rad
	double sibry; // Poloidal flux at the plasma boundary in Weber / rad
	int idum;
	double xdum;

	char str_buff[50];

	inFileStream_.get(str_buff, 48);

	desc = std::string(str_buff);

	inFileStream_ >> std::setw(4) >> idum >> nw >> nh;

	inFileStream_ >> std::setw(16)

	>> rdim >> zdim >> rcentr >> rleft >> zmid

	>> rmaxis >> zmaxis >> simag >> sibry >> bcentr

	>> current >> simag >> xdum >> rmaxis >> xdum

	>> zmaxis >> xdum >> sibry >> xdum >> xdum;

	rzmin_[0] = rleft;
	rzmax_[0] = rleft + rdim;

	rzmin_[1] = zmid - zdim / 2;
	rzmax_[1] = zmid + zdim / 2;

	dims_[0] = nw;
	dims_[1] = nh;

	inter2d_type(dims_, rzmin_, rzmax_).swap(psirz_);

#define INPUT_VALUE(_NAME_)                                                            \
	for (int s = 0; s < nw; ++s)                                              \
	{                                                                                  \
		value_type y;                                                                  \
		inFileStream_ >> std::setw(16) >> y;                                           \
		profile_[ _NAME_ ].data().emplace(                                                         \
	     std::make_pair(static_cast<value_type>(s)                                     \
	          /static_cast<value_type>(nw-1), y));                               \
	}                                                                                  \

	INPUT_VALUE("fpol");
	INPUT_VALUE("pres");
	INPUT_VALUE("ffprim");
	INPUT_VALUE("pprim");

	for (int j = 0; j < nh; ++j)
		for (int i = 0; i < nw; ++i)
		{
			value_type v;
			inFileStream_ >> std::setw(16) >> v;
			psirz_[i + j * nw] = (v - simag) / (sibry - simag); // Normalize Poloidal flux
		}

	INPUT_VALUE("qpsi");

#undef INPUT_VALUE

	unsigned int nbbbs, limitr;
	inFileStream_ >> std::setw(5) >> nbbbs >> limitr;

	rzbbb_.resize(nbbbs);
	rzlim_.resize(limitr);
	inFileStream_ >> std::setw(16) >> rzbbb_;
	inFileStream_ >> std::setw(16) >> rzlim_;

	ReadProfile(fname + "_profiles.txt");

}
void GEqdsk::ReadProfile(std::string const &fname)
{
	LOGGER << "Load GFile Profiles: " << fname;
	std::ifstream inFileStream_(fname);

	if (!inFileStream_.is_open())
	{
		ERROR << "File " << fname << " is not opend!";
		return;
	}

	std::string line;

	std::getline(inFileStream_, line);

	std::vector<std::string> names;
	{
		std::stringstream lineStream(line);

		while (lineStream)
		{
			std::string t;
			lineStream >> t;
			if (t != "")
				names.push_back(t);
		};
	}

	while (inFileStream_)
	{
		auto it = names.begin();
		auto ie = names.end();
		Real psi;
		inFileStream_ >> psi; 		/// @NOTE assume first row is psi
		*it = psi;

		for (++it; it != ie; ++it)
		{
			Real value;
			inFileStream_ >> value;
			profile_[*it].data().emplace(psi, value);

		}
	}
}

void GEqdsk::Save(std::ostream & os) const
{
	os << Dump(psirz_.data(), "psi", 2, &dims_[0]) << std::endl;

	size_t num = rzbbb_.size();

	os << Dump(&rzbbb_[0], "rzbbb", 1, &num) << std::endl;

	num = rzlim_.size();

	os << Dump(rzlim_, "rzlim") << std::endl;

	for (auto const & p : profile_)
	{
		os << Dump(p.second.data(), p.first) << std::endl;
	}
}
std::ostream & GEqdsk::Print(std::ostream & os)
{
	std::cout << "--" << desc << std::endl;

//	std::cout << "nw" << "\t= " << nw
//			<< "\t--  Number of horizontal R grid  points" << std::endl;
//
//	std::cout << "nh" << "\t= " << nh << "\t-- Number of vertical Z grid points"
//			<< std::endl;
//
//	std::cout << "rdim" << "\t= " << rdim
//			<< "\t-- Horizontal dimension in meter of computational box                   "
//			<< std::endl;
//
//	std::cout << "zdim" << "\t= " << zdim
//			<< "\t-- Vertical dimension in meter of computational box                   "
//			<< std::endl;

	std::cout << "rcentr" << "\t= " << rcentr
	        << "\t--                                                                    " << std::endl;

//	std::cout << "rleft" << "\t= " << rleft
//			<< "\t-- Minimum R in meter of rectangular computational box                "
//			<< std::endl;
//
//	std::cout << "zmid" << "\t= " << zmid
//			<< "\t-- Z of center of computational box in meter                          "
//			<< std::endl;

	std::cout << "rmaxis" << "\t= " << rmaxis
	        << "\t-- R of magnetic axis in meter                                        " << std::endl;

	std::cout << "rmaxis" << "\t= " << zmaxis
	        << "\t-- Z of magnetic axis in meter                                        " << std::endl;

//	std::cout << "simag" << "\t= " << simag
//			<< "\t-- poloidal flus ax magnetic axis in Weber / rad                      "
//			<< std::endl;
//
//	std::cout << "sibry" << "\t= " << sibry
//			<< "\t-- Poloidal flux at the plasma boundary in Weber / rad                "
//			<< std::endl;

	std::cout << "rcentr" << "\t= " << rcentr
	        << "\t-- R in meter of  vacuum toroidal magnetic field BCENTR               " << std::endl;

	std::cout << "bcentr" << "\t= " << bcentr
	        << "\t-- Vacuum toroidal magnetic field in Tesla at RCENTR                  " << std::endl;

	std::cout << "current" << "\t= " << current
	        << "\t-- Plasma current in Ampere                                          " << std::endl;

//	std::cout << "fpol" << "\t= "
//			<< "\t-- Poloidal current function in m-T<< $F=RB_T$ on flux grid           "
//			<< std::endl << fpol_.data() << std::endl;
//
//	std::cout << "pres" << "\t= "
//			<< "\t-- Plasma pressure in $nt/m^2$ on uniform flux grid                   "
//			<< std::endl << pres_.data() << std::endl;
//
//	std::cout << "ffprim" << "\t= "
//			<< "\t-- $FF^\\prime(\\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid     "
//			<< std::endl << ffprim_.data() << std::endl;
//
//	std::cout << "pprim" << "\t= "
//			<< "\t-- $P^\\prime(\\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid    "
//			<< std::endl << pprim_.data() << std::endl;
//
//	std::cout << "psizr"
//			<< "\t-- Poloidal flus in Webber/rad on the rectangular grid points         "
//			<< std::endl << psirz_.data() << std::endl;
//
//	std::cout << "qpsi" << "\t= "
//			<< "\t-- q values on uniform flux grid from axis to boundary                "
//			<< std::endl << qpsi_.data() << std::endl;
//
//	std::cout << "nbbbs" << "\t= " << nbbbs
//			<< "\t-- Number of boundary points                                          "
//			<< std::endl;
//
//	std::cout << "limitr" << "\t= " << limitr
//			<< "\t-- Number of limiter points                                           "
//			<< std::endl;
//
//	std::cout << "rzbbbs" << "\t= "
//			<< "\t-- R of boundary points in meter                                      "
//			<< std::endl << rzbbb_ << std::endl;
//
//	std::cout << "rzlim" << "\t= "
//			<< "\t-- R of surrounding limiter contour in meter                          "
//			<< std::endl << rzlim_ << std::endl;

	return os;
}

}  // namespace simpla

