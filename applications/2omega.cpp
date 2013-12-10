/*
 * 2omega.cpp
 *
 *  Created on: 2013年12月10日
 *      Author: salmon
 */

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "../src/simpla_defs.h"
#include "../src/fetl/fetl.h"
#include "../src/mesh/co_rect_mesh.h"
#include "../src/utilities/log.h"

using namespace simpla;

struct Context2Omega
{
	typedef CoRectMesh<Complex> Mesh;

	typedef typename Mesh::scalar_type scalar_type;

	typedef Field<Geometry<Mesh, 0>, scalar_type> ZeroForm;
	typedef Field<Geometry<Mesh, 1>, scalar_type> OneForm;
	typedef Field<Geometry<Mesh, 2>, scalar_type> TwoForm;

	Context2Omega(int argc, char **argv) :
			E(mesh), B(mesh), J(mesh),

			k_parallel(0.0), num_of_step(100), record_stride(1)
	{

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
				Log::OpenFile(value);
				break;
			case 'v':
				Log::Verbose(atof(value));
				break;
			case 'h':
				help_mesage();
				exit(1);
				break;
			default:
				std::cout << SIMPLA_LOGO << std::endl;

			}

		}

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

		mu0 = mesh.constants["permeability of free space"];
		epsilon0 = mesh.constants["permittivity of free space"];
		speed_of_light = mesh.constants["speed of light"];
		proton_mass = mesh.constants["proton mass"];
		elementary_charge = mesh.constants["elementary charge"];

		E = 0.0;
		B = 0.0;
		J = 0.0;

	}

	~Context2Omega()
	{
	}

public:

	Mesh mesh;

	size_t num_of_step;
	size_t record_stride;
	std::string workspace_path;

	Real dt;
	Real k_parallel;
	size_t num_points;

	double mu0;
	double epsilon0;
	double speed_of_light;
	double proton_mass;
	double elementary_charge;

	OneForm E;
	OneForm J;
	TwoForm B;
};

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

	std::cout << SIMPLA_LOGO << std::endl;

	Log::Verbose(0);

	Context2Omega ctx(argc, argv);

//  Summary    ====================================

	std::cout << std::endl << DOUBLELINE << std::endl;

	std::cout << "[Main Control]" << std::endl;

	std::cout << SINGLELINE << std::endl;

//	mesh.Print(std::cout);

	std::cout << SINGLELINE << std::endl;

}
