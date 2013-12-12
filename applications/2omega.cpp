/*
 * 2omega.cpp
 *
 *  Created on: 2013年12月10日
 *      Author: salmon
 */

#include <complex>
#include "../src/simpla_defs.h"
#include "../src/fetl/fetl.h"
#include "../src/mesh/co_rect_mesh.h"
#include "../src/particle/particle.h"
#include "pic/ggauge.h"

namespace simpla
{

template<typename TM>
struct Context2Omega
{
public:

	typedef TM mesh_type;
	typedef typename mesh_type::scalar scalar;

	DEFINE_FIELDS(TM)

	Particle<mesh_type, GGauge<mesh_type, 16> > ion;
	Particle<mesh_type, GGauge<mesh_type, 4> > electron;

public:

	Context2Omega(int argc, char **argv);
	~Context2Omega();
	inline void OneStep();

public:
	mesh_type mesh;

	Form<1> E;
	Form<1> J;
	Form<2> B;
	RVectorForm<0> B0;
}
;

template<typename TM>
Context2Omega<TM>::Context2Omega(int argc, char **argv) :
		E(mesh), B(mesh), J(mesh), B0(mesh), ion(mesh), electron(mesh)
{

	Real dt;
	Real k_parallel;
	size_t num_points;

	mesh.dt_ = dt;

	mesh.xmin_[0] = 0;
	mesh.xmin_[1] = 0;
	mesh.xmin_[2] = 0;
	mesh.xmax_[0] = 1.0;
	mesh.xmax_[1] = 1.0 / k_parallel;
	mesh.xmax_[2] = 0;
	mesh.dims_[0] = num_points;
	mesh.dims_[1] = 1;
	mesh.dims_[2] = 1;
	mesh.gw_[0] = 2;
	mesh.gw_[1] = 0;
	mesh.gw_[2] = 0;

	mesh.Update();

	E = 0.0;
	B = 0.0;
	J = 0.0;
	B0 = 0.0;

}

template<typename TM>
Context2Omega<TM>::~Context2Omega()
{
}

template<typename TM>
inline void Context2Omega<TM>::OneStep()
{
	const double mu0 = mesh.constants["permeability of free space"];
	const double epsilon0 = mesh.constants["permittivity of free space"];
	//	const double speed_of_light = mesh.constants["speed of light"];
	//	const double proton_mass = mesh.constants["proton mass"];
	//	const double elementary_charge = mesh.constants["elementary charge"];

	E += (Curl(B / mu0) - J) / epsilon0 * mesh.GetDt();
	B -= Curl(E) * (mesh.GetDt() * 0.5);

	ion.Push(B0, E, B);
	electron.Push(B0, E, B);

	ion.Collect(&J, B0, E, B);
	electron.Collect(&J, B0, E, B);
	B -= Curl(E) * (mesh.GetDt() * 0.5);
}
}
// namespace simpla

void help_mesage()
{
	std::cout << "Too lazy to write a complete help information\n"
			"\t -n<NUM>\t number of steps\n"
			"\t -s<NUM>\t recorder per <NUM> steps\n"
			"\t -o<STRING>\t output directory\n"
			"\t -i<STRING>\t configure file "
			"\n" << std::endl;
}

int main(int argc, char **argv)
{

	size_t num_of_step;

	size_t record_stride;

	std::string workspace_path;

	if (argc <= 1)
	{
		help_mesage();
		exit(1);
	}

	for (int i = 1; i < argc; ++i)
	{
		char opt = *(argv[i] + 1);
		char * value = argv[i] + 2;

		switch (opt)
		{
		case 'n':
			num_of_step = atoi(value);
			break;
		case 's':
			record_stride = atoi(value);
			break;
		case 'o':
			workspace_path = value;
			break;
		case 'l':
			Logger::OpenFile(value);
			break;
		case 'v':
			Logger::Verbose(atof(value));
			break;
		case 'h':
			help_mesage();
			exit(1);
			break;
		default:
			std::cout << SIMPLA_LOGO << std::endl;

		}

	}

	std::cout << SIMPLA_LOGO << std::endl;

	Logger::Verbose(0);

	simpla::Context2Omega<simpla::CoRectMesh<std::complex<double>> > ctx(argc,
			argv);

//  Summary    ====================================

	std::cout << std::endl << DOUBLELINE << std::endl;

	std::cout << "[Main Control]" << std::endl;

	std::cout << SINGLELINE << std::endl;

//	mesh.Print(std::cout);

	std::cout << SINGLELINE << std::endl;

}
