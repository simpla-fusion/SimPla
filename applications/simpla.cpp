/*
 * simpla.cpp
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "../src/simpla_defs.h"
#include "../src/fetl/fetl.h"
#include "../src/particle/particle.h"
#include "../src/particle/pic_engine_default.h"
#include "../src/particle/pic_engine_deltaf.h"
#include "../src/utilities/lua_state.h"
#include "../src/mesh/co_rect_mesh.h"
#include "../src/engine/basecontext.h"
#include "pic/pic_engine_ggauge.h"
#include "solver/electromagnetic/cold_fluid.h"

namespace simpla
{

template<typename TM>
struct Context: public BaseContext
{
public:
	typedef BaseContext base_type;
	typedef TM mesh_type;
	typedef typename mesh_type::scalar scalar;
	typedef LuaObject configure_type;

	DEFINE_FIELDS(TM)
public:

	Context();
	~Context();
	void Deserialize(configure_type const & cfg);
	void Serialize(configure_type * cfg) const;
	void OneStep();
	void DumpData() const;
	inline ParticleCollection<mesh_type> &
	GetParticleCollection()
	{
		return particle_collection_;
	}
public:
	mesh_type mesh;

	Form<1> E1;
	Form<1> J1;
	Form<2> B1;
	RVectorForm<0> B0;
	RForm<0> n0;

	ColdFluidEM<mesh_type> cold_fluid_;
	ParticleCollection<mesh_type> particle_collection_;

	typedef typename ParticleCollection<mesh_type>::particle_type particle_type;
}
;

template<typename TM>
Context<TM>::Context() :
		E1(mesh), B1(mesh), J1(mesh), B0(mesh), n0(mesh),

		cold_fluid_(mesh), particle_collection_(mesh)
{

	particle_collection_.template RegisterFactory<GGauge<mesh_type, 0>>(
			"GuidingCenter");
	particle_collection_.template RegisterFactory<GGauge<mesh_type, 8>>(
			"GGauge8");
	particle_collection_.template RegisterFactory<GGauge<mesh_type, 32>>(
			"GGauge32");
	particle_collection_.template RegisterFactory<PICEngineDefault<mesh_type> >(
			"Default");
	particle_collection_.template RegisterFactory<PICEngineDeltaF<mesh_type> >(
			"DeltaF");
}

template<typename TM>
Context<TM>::~Context()
{
}
template<typename TM>
void Context<TM>::Deserialize(LuaObject const & cfg)
{

	mesh.Deserialize(cfg.GetChild("Grid"));

	cold_fluid_.Deserialize(cfg.GetChild("Particles"));

	particle_collection_.Deserialize(cfg.GetChild("Particles"));

	LoadField(cfg.GetChild("n0"), &n0);
	LoadField(cfg.GetChild("B0"), &B0);
	LoadField(cfg.GetChild("E1"), &E1);
	LoadField(cfg.GetChild("B1"), &B1);
	LoadField(cfg.GetChild("E1"), &J1);
}

template<typename TM>
void Context<TM>::Serialize(configure_type * cfg) const
{
}

template<typename TM>
void Context<TM>::OneStep()
{
	base_type::OneStep();
}

template<typename TM>
void Context<TM>::DumpData() const
{
}

}  // namespace simpla

using namespace simpla;

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

	LuaObject pt;

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
		case 'i':
			pt.ParseFile(value);
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
	std::shared_ptr<BaseContext> ctx;
	auto grid = pt.GetChild("Grid");
	if (grid.at("Type").as<std::string>() == "CoRectMesh")
	{
		typedef CoRectMesh<Complex> mesh_type;
		std::shared_ptr<Context<mesh_type>> ctx_ptr(new Context<mesh_type>);

		ctx = std::dynamic_pointer_cast<BaseContext>(ctx_ptr);
	}

//  Summary    ====================================

	std::cout << std::endl << DOUBLELINE << std::endl;

	std::cout << "[Main Control]" << std::endl;

	std::cout << SINGLELINE << std::endl;

//	mesh.Print(std::cout);

	std::cout << SINGLELINE << std::endl;

// Main Loop ============================================

	INFORM << (">>> Pre-Process DONE! <<<");
	ctx->Deserialize(pt);
	INFORM << (">>> Process START! <<<");

	for (int i = 0; i < num_of_step; ++i)
	{
		INFORM << ">>> STEP " << i << " Start <<<";

		ctx->OneStep();

		if (i % record_stride == 0)
		{

		}
		INFORM << ">>> STEP " << i << " Done <<<";
	}

	INFORM << (">>> Process DONE! <<<");
	LuaObject dump;
	ctx->Serialize(&dump);
	INFORM << (">>> Post-Process DONE! <<<");
////
////// Log ============================================

}
