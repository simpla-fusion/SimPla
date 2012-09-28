/*
 * possion.cpp
 *
 *  Created on: 2012-3-30
 *      Author: salmon
 */
#include <string>
#include <iostream>
#include "include/simpla_defs.h"
#include "primitives/ntuple.h"
#include "fetl/fetl.h"
#include "engine/context.h"
#include "io/io.h"
#include "solvers/linear_solver/ksp_cg.h"
#include "utilities/log.h"

using namespace simpla;
using namespace simpla::fetl;
int main(int argc, char **argv)
{
	Log::info_level = 0;

	IVec3 dims =
	{ 201, 101, 1 };

	Vec3 xmin =
	{ 0, 0, 0 };

	Vec3 xmax =
	{ 20, 20, 1 };

	size_t max_step = 1000;

	size_t record_stride = 1;

	std::string output = "untitle";

	double omega = 1.0;

	double JAmp = 1.0;

	double N0 = 1.0;

	double T = 0.001;
	size_t iter_num = 1000;
	double noise = 0;
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
		case 'f':
			omega = atof(argv[i] + 2);
			break;
		case 'J':
			JAmp = atof(argv[i] + 2);
			break;
		case 'L':
			xmax[0] = atof(argv[i] + 2);
			break;
		case 'N':
			N0 = atof(argv[i] + 2);
			break;
		case 'T':
			T = atof(argv[i] + 2);
			break;
		case 'v':
			Log::Verbose(atof(argv[i] + 2));
			break;
		case 'i':
			iter_num = atoi(argv[i] + 2);
			break;
		case 'd':
			noise = atof(argv[i] + 2);
			break;
		}

	}

	INFORM << ">>> START ! <<<";

	typedef DEFAULT_GRID Grid;
	TR1::shared_ptr<Context<Grid> > ctx(new Context<Grid>());
	ctx->grid.Initialize(1 / 200.0, xmin, xmax, dims);

	Grid const & grid = ctx->grid;

	INFORM << "==========================================================";
	INFORM << "Record \t:" << max_step << "/" << record_stride;
	INFORM << "dims \t:" << grid.dims;
	INFORM << "xmin \t:" << grid.xmin;
	INFORM << "xmax \t:" << grid.xmax;
	INFORM << "dx \t:" << grid.dx;
	INFORM << "inv_dx \t:" << grid.inv_dx;
	INFORM << "dt \t:" << grid.dt;
	INFORM << "J Amp. \t:" << JAmp;
	INFORM << "Omega. \t:" << omega;
	INFORM << "T \t:" << T;
	INFORM << "==========================================================";

	simpla::io::IOEngine<Grid> diag(ctx, output);

	ZeroForm & rho = *(ctx->GetObject<ZeroForm>("rho"));
	ZeroForm & phi = *(ctx->GetObject<ZeroForm>("phi"));
	ZeroForm & res = *(ctx->GetObject<ZeroForm>("res"));

	rho = 0;

	double k0 = 1.0 * 2.0 * 3.141592653589793 / (xmax[0] - xmin[0]);
	double k1 = 2.0 * 2.0 * 3.141592653589793 / (xmax[1] - xmin[1]);

#pragma omp parallel for
	for (size_t i = 0; i < grid.dims[0]; ++i)
		for (size_t j = 0; j < grid.dims[1]; ++j)
			for (size_t k = 0; k < grid.dims[2]; ++k)
			{
				size_t s = grid.get_cell_num(i, j, k);
				rho[s] = sin(k0 * i * grid.dx[0]) * sin(k1 * j * grid.dx[1]);
			}

	ctx->PreProcess();

	diag.Register("rho", record_stride);

	diag.Register("phi", record_stride);

	diag.Register("res", record_stride);

	linear_solver::ksp_cg_solver(Diverge(Grad(phi)) + rho, phi, iter_num);

	rho = rho * 2.0;

	linear_solver::ksp_cg_solver(Diverge(Grad(phi)) == -rho, phi, iter_num);

	res = Diverge(Grad(phi)) + rho;

	diag.WriteAll();

	ctx->PostProcess();

	INFORM << ">>> DONE ! <<<";
}

