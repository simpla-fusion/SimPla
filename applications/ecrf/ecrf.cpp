/*
 * ecrf.cpp
 *
 *  Created on: 2012-3-6
 *      Author: salmon
 */

#include "include/simpla_defs.h"
#include "physics/constants.h"
#include "fetl/fetl.h"
#include "utilities/properties.h"

#include "engine/modules.h"

#include "modules/em/maxwell.h"
#include "modules/em/pml.h"
#include "modules/pic/pic.h"
#include "modules/fluid/cold_fluid.h"

//#include "io/io.h"
using namespace simpla;

DEFINE_FIELDS(Real, UniformRectGrid);

int main(int argc, char **argv)
{
	Log::info_level = 0;

	size_t max_step = 1000;

	size_t record_stride = 1;

	std::string input = "simpla.xml";
	std::string output = "untitle";

	double omega = 1.0;

	double JAmp = 1.0;

	double N0 = 1.0;

	double T = 0.001;

	for (int i = 1; i < argc; ++i)
	{
		switch (argv[i][1])
		{
		case 'n':
			max_step = atoi(argv[i] + 2);
			break;
		case 's':
			record_stride = atoi(argv[i] + 2);
			break;
		case 'o':
			output = argv[i] + 2;
			break;
		case 'i':
			input = argv[i] + 2;
			break;
		case 'v':
			Log::Verbose(atof(argv[i] + 2));
			break;
		}

	}

	boost::property_tree::ptree pt;

	read_file(input, pt);

	Domain domain(pt.get_child("Domain"));

//  std::cout << "=========================================================="
//      << std::endl;
//  std::cout << "Record \t:" << max_step << "/" << record_stride << std::endl;
//  std::cout << "dims \t:" << grid->dims << std::endl;
//  std::cout << "xmin \t:" << grid->xmin << std::endl;
//  std::cout << "xmax \t:" << grid->xmax << std::endl;
//  std::cout << "dt \t:" << grid->dt << std::endl;
//  std::cout << "J Amp. \t:" << JAmp << std::endl;
//  std::cout << "Omega. \t:" << omega << std::endl;
//  std::cout << "T \t:" << T << std::endl;
//  std::cout << "=========================================================="
//      << std::endl;
//
////	simpla::io::IOEngine<Grid> diag(domain, output);
//
//	ZeroForm &n1 = *domain.GetObject<ZeroForm>("n0");
//
//	for (size_t s = 0; s < dims[0]; ++s)
//	{
//		n1[s] = N0 * 0.5
//				* (1.0
//						- std::cos(
//								PI * static_cast<double>(s)
//										/ static_cast<double>(dims[0] - 1)));
//	}
//
////	domain.AddSolver("DeltaF", new PICEngine<DeltaF, Grid>(domain));

//	if (boost::optional<ptree &> module = pt.get_child_optional("Modules.PML"))
//	{
//		domain.functions.push_back(
//				TR1::bind(&em::PML<Real, UniformRectGrid>::Eval,
//						new em::PML<Real, Grid>(domain, *module)));
//	}

	if (boost::optional<ptree &> module = pt.get_child_optional(
			"Modules.Maxwell"))
	{
		domain.functions.push_back(
				TR1::bind(&em::Maxwell<Real, Grid>::Eval,
						new em::Maxwell<Real, Grid>(domain, *module)));
	}

	if (boost::optional<ptree &> module = pt.get_child_optional(
			"Modules.ColdFluid"))
	{
		domain.functions.push_back(
				TR1::bind(&em::ColdFluid<Real, UniformRectGrid>::Eval,
						new em::ColdFluid<Real, Grid>(domain, *module)));
	}

//	domain.PreProcess();

//	diag.Register("E1", record_stride);
//	diag.Register("B1", record_stride);
//	diag.Register("J1", record_stride);
//	diag.Register("n1", record_stride);
//	diag.Register("electron", record_stride);
//
//	for (size_t s = 0; s < max_step; ++s)
//	{
////		diag.WriteAll();
//		n1 = 0.0;
//
////		J1[6 * 3 + 1] = JAmp * sin(omega * domain.Time());
//
//		domain.Eval();
//
//	}

//	domain.PostProcess();

}

