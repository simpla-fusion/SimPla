/*
 * ecrf.cpp
 *
 *  Created on: 2012-3-6
 *      Author: salmon
 */

#include "include/simpla_defs.h"
#include "physics/constants.h"
#include "fetl/fetl.h"
#include "engine/context.h"
#include "grid/uniform_rect.h"
//#include "io/io.h"
#include "primitives/properties.h"
#include "solvers/em/pml.h"
#include "solvers/em/maxwell.h"
#include "pic/pic.h"
#include "pic/full_f.h"

using namespace simpla;
using namespace simpla::em;
using namespace simpla::pic;
using namespace simpla::fetl;

DEFINE_FIELDS(Real, UniformRectGrid);

int
main(int argc, char **argv)
{
  Log::info_level = 0;

  IVec3 dims =
    { 200, 1, 1 };

  Vec3 xmin =
    { 0, 0, 0 };

  Vec3 xmax =
    { 20, 1, 1 };

  size_t max_step = 1000;

  size_t record_stride = 1;

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

        }

    }

  boost::property_tree::ptree pt;

  TR1::shared_ptr<Grid> grid(new Grid(pt));

  TR1::shared_ptr<Context> ctx(new Context(grid,pt));


  std::cout << "=========================================================="
      << std::endl;
  std::cout << "Record \t:" << max_step << "/" << record_stride << std::endl;
  std::cout << "dims \t:" << grid->dims << std::endl;
  std::cout << "xmin \t:" << grid->xmin << std::endl;
  std::cout << "xmax \t:" << grid->xmax << std::endl;
  std::cout << "dt \t:" << grid->dt << std::endl;
  std::cout << "J Amp. \t:" << JAmp << std::endl;
  std::cout << "Omega. \t:" << omega << std::endl;
  std::cout << "T \t:" << T << std::endl;
  std::cout << "=========================================================="
      << std::endl;

//	simpla::io::IOEngine<Grid> diag(ctx, output);


  ZeroForm & n1 = *(ctx->GetObject<ZeroForm>("n1"));
  OneForm & J1 = *(ctx->GetObject<OneForm>("J1"));
  OneForm & E1 = *(ctx->GetObject<OneForm>("E1"));
  TwoForm & B1 = *(ctx->GetObject<TwoForm>("B1"));
  TwoForm & B0 = *(ctx->GetObject<TwoForm>("B0"));

  std::cout << n1.get_element_size_in_bytes() << std::endl;

  n1 = 1.0;
  J1 = 0.0;
  E1 = 0.0;
  B1 = 0.0;
  Vec3 b0 =
    { 0.0, 0.0, 1.0 };
  B0 = b0;

  for (size_t s = 0; s < dims[0]; ++s)
    {
      n1[s] = N0 * 0.5
          * (1.0
              - std::cos(
                  PI * static_cast<double>(s)
                      / static_cast<double>(dims[0] - 1)));
    }

  nTuple<SIX, int> bc =
    { 5, 5, 0, 0, 0, 0 };

//	ctx->AddSolver("DeltaF", new PICEngine<DeltaF, Grid>(ctx));



  ctx->functions.push_back(
      TR1::bind(&pic::PICEngine<pic::FullF, Grid>::Process,
          pic::PICEngine<pic::FullF, Grid>::Create(ctx, pt)));

  ctx->functions.push_back(TR1::bind(&Maxwell<Real, Grid>, ctx));

  ctx->functions.push_back(
      TR1::bind(&PML<Real, Grid>::Process, new PML<Real, Grid>(ctx, pt)));

//	ctx->PreProcess();

//	diag.Register("E1", record_stride);
//	diag.Register("B1", record_stride);
//	diag.Register("J1", record_stride);
//	diag.Register("n1", record_stride);
//	diag.Register("electron", record_stride);

  for (size_t s = 0; s < max_step; ++s)
    {
//		diag.WriteAll();
      n1 = 0.0;
      J1 = 0.0;

      J1[6 * 3 + 1] = JAmp * sin(omega * ctx->Time());

      ctx->Eval();

    }

//	ctx->PostProcess();

}

