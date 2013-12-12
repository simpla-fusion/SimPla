/*
 * simpla.cpp
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <new>
#include <string>

#include "../src/simpla_defs.h"

#include "../src/engine/basecontext.h"
#include "../src/fetl/fetl.h"
#include "../src/io/data_stream.h"

#include "../src/mesh/co_rect_mesh.h"

#include "../src/particle/particle.h"
#include "../src/particle/pic_engine_default.h"
#include "../src/particle/pic_engine_deltaf.h"

#include "../src/utilities/log.h"
#include "../src/utilities/lua_state.h"
#include "../src/utilities/parse_command_line.h"
#include "../src/utilities/utilities.h"

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
	typedef Context<TM> this_type;

	template<typename U>
	friend std::ostream & operator<<(std::ostream & os, Context<U> const &self);

	Context();
	~Context();
	void Deserialize(configure_type const & cfg);
	void Serialize(configure_type * cfg) const;
	void OneStep();
	void DumpData(std::string const &) const;

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

	mesh.Deserialize(cfg["Grid"]);

	cold_fluid_.Deserialize(cfg["Particles"]);

	LOG << " Load Cold Fluid [Done]!";

	particle_collection_.Deserialize(cfg["Particles"]);

	LOG << " Load Particles [Done]!";

	auto init_value = cfg["InitValue"];

	n0.Init();
	LoadField(init_value["n0"], &n0);
	B0.Init();
	LoadField(init_value["B0"], &B0);
	E1.Init();
	LoadField(init_value["E1"], &E1);
	B1.Init();
	LoadField(init_value["B1"], &B1);
	J1.Init();
	LoadField(init_value["J1"], &J1);

	LOG << " Load Initial Fields [Done]!";
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
void Context<TM>::DumpData(std::string const &) const
{
}
template<typename TM> std::ostream & operator<<(std::ostream & os,
		Context<TM> const &self)

{

	os

	<< self.mesh << "\n"

	<< self.cold_fluid_ << "\n"

	<< self.particle_collection_ << "\n"

	<< "InitValue={" << "\n"

	<< " n0=" << Data(self.n0, "n0") << ",\n"

	<< " E1=" << Data(self.E1, "E1") << ",\n"

	<< " B1=" << Data(self.B1, "B1") << ",\n"

	<< " J1=" << Data(self.J1, "J1") << ",\n"

	<< " B0=" << Data(self.B0, "B0") << "\n"

	<< "}" << "\n"

	;

	return os;
}
}  // namespace simpla

using namespace simpla;

int main(int argc, char **argv)
{

	Log::Verbose(10);

	LuaObject pt;

	size_t num_of_step;

	size_t record_stride;

	std::string work_path = "./";

	bool just_a_test = false;

	ParseCmdLine(argc, argv,
			[&](std::string const & opt,std::string const & value)->int
			{
				if(opt=="n"||opt=="num_of_step")
				{
					num_of_step =ToValue<size_t>(value);
				}
				else if(opt=="s"||opt=="record_stride")
				{
					record_stride =ToValue<size_t>(value);
				}
				else if(opt=="o"||opt=="output"||opt=="p"||opt=="prefix")
				{
					GLOBAL_DATA_STREAM.OpenFile(value);
				}
				else if(opt=="i"||opt=="input")
				{
					pt.ParseFile(value);
				}
				else if(opt=="c"|| opt=="command")
				{
					pt.ParseString(value);
				}
				else if(opt=="l"|| opt=="log")
				{
					Log::OpenFile (value);
				}
				else if(opt=="v"|| opt=="verbose")
				{
					Log::Verbose(ToValue<int>(value));
				}
				else if(opt=="q"|| opt=="quiet")
				{
					Log::Verbose(ToValue<int>(value));
				}
				else if(opt=="g"|| opt=="generator")
				{
					INFORM
					<< ShowCopyRight() << std::endl
					<< "Too lazy to implemented it\n"<< std::endl;
					exit(1);
				}
				else if(opt=="t")
				{
					just_a_test=true;
				}
				else if(opt=="V")
				{
					INFORM<<ShowShortVersion()<< std::endl;
					exit(1);
				}

				else if(opt=="version")
				{
					INFORM<<ShowVersion()<< std::endl;
					exit(1);
				}
				else if(opt=="help")
				{
					INFORM
					<< ShowCopyRight() << std::endl
					<< "Too lazy to write a complete help information\n"<< std::endl;
					exit(1);

				}
				else
				{
					INFORM
					<< ShowCopyRight() << std::endl
					<<
					" -h        \t print this information\n"
					" -n<NUM>   \t number of steps\n"
					" -s<NUM>   \t recorder per <NUM> steps\n"
					" -o<STRING>\t output directory\n"
					" -i<STRING>\t configure file \n"
					" -c,--config <STRING>\t Lua script passed in as string \n"
					" -t        \t only read and parse input file, but do not process  \n"
					" -g,--generator   \t generator a demo input script file \n"
					" -v<NUM>   \t verbose  \n"
					" -V        \t print version  \n"
					" -q        \t quiet mode, standard out  \n"
					;
					exit(1);
				}
				return CONTINUE;

			}

			);

	INFORM << SIMPLA_LOGO << std::endl;

	LOG << "Parse Command Line: [Done]!";

	if (pt.isNull())
	{
		LOG << "Nothing to do !!";
		exit(1);
	}

	std::shared_ptr<BaseContext> ctx;

//	try
//	{
	auto mesh_type = pt.GetChild("Grid").GetChild("Type").as<std::string>();

	if (mesh_type == "CoRectMesh")
	{
		typedef CoRectMesh<Complex> mesh_type;
		std::shared_ptr<Context<mesh_type>> ctx_ptr(new Context<mesh_type>);
		ctx = std::dynamic_pointer_cast<BaseContext>(ctx_ptr);
		ctx->Deserialize(pt);

		std::cout << (*ctx_ptr);

	}
//	} catch (...)
//	{
////		pt.Serialize(std::cout);
//		ERROR << "Configure Error!";
//	}

//  Summary    ====================================

	LOG << std::endl << DOUBLELINE << std::endl;

	LOG << "[Main Control]" << std::endl;

// Main Loop ============================================

	LOG << std::endl << DOUBLELINE << std::endl;
	LOG << (">>> Pre-Process [Done]! <<<");

	if (!just_a_test)
	{
		LOG << (">>> Process [Start]! <<<");
		for (int i = 0; i < num_of_step; ++i)
		{
			LOG << ">>> STEP " << i << " Start <<<";

			ctx->OneStep();

			if (i % record_stride == 0)
			{
				ctx->DumpData(work_path);
			}
			LOG << ">>> STEP " << i << " [Done] <<<";
		}
		LOG << (">>> Process [Done]! <<<");
		//	ctx->Serialize(std::cout);
		LOG << (">>> Post-Process [Done]! <<<");
	}

////
////// Log ============================================

}
