/*
 * ecrf.cpp
 *
 *  Created on: 2012-3-6
 *      Author: salmon
 */

#include "include/simpla_defs.h"
#include "physics/constants.h"
#include "fetl/fetl.h"
//#include "pic/pic.h"
#include "engine/context.h"
#include "engine/context_impl.h"
#include "fetl/grid/uniform_rect.h"

//#include "io/io.h"
using namespace simpla;

DEFINE_FIELDS(Real, UniformRectGrid);

int main(int argc, char **argv)
{
	Log::info_level = 0;

	size_t max_step = 1000;

	size_t record_stride = 1;

	std::string input = "simpla.xml";
	std::string output = "untitle.info";

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

	write_file(output, pt);

	Context<UniformRectGrid> ctx(pt.get_child("Context"));

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


	std::cout

	<< SIMPLA_LOGO << std::endl

	<< std::setw(20) << "Teimstamp : " << Log::Teimstamp() << std::endl

	<< std::setw(20) << "Num. of procs. : " << omp_get_num_procs() << std::endl

	<< std::setw(20) << "Num. of threads : " << omp_get_max_threads()

	<< std::endl

	<< std::setw(20) << "Configure File : " << input << std::endl

	<< std::setw(20) << "Output Path : " << output << std::endl

	<< std::setw(20) << "Number of steps : " << max_step << std::endl

	<< std::setw(20) << "Record/steps : " << record_stride << std::endl

	<< SINGLELINE << std::endl

	<< ctx.Summary() << std::endl

	<< SINGLELINE << std::endl

	<< std::endl;

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

