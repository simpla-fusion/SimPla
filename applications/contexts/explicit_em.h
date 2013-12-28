/*
 * \file explicit_em.h
 *
 *  Created on: 2013年12月12日
 *      Author: salmon
 */

#ifndef EXPLICIT_EM_H_
#define EXPLICIT_EM_H_

#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
//#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <valarray>
#include <vector>
#include <map>
#include <unordered_map>

#include "../../src/engine/basecontext.h"

#include "../../src/fetl/fetl.h"
#include "../../src/fetl/load_field.h"
#include "../../src/fetl/save_field.h"

#include "../../src/mesh/media_tag.h"

#include "../../src/fetl/field_function.h"

#include "../../src/utilities/log.h"
#include "../../src/utilities/lua_state.h"
#include "../../src/io/data_stream.h"

#include "../pic/pic_engine_ggauge.h"
#include "../../src/particle/particle.h"
#include "../../src/particle/pic_engine_full.h"
//#include "../../src/particle/pic_engine_deltaf.h"

#include "../solver/electromagnetic/cold_fluid.h"
#include "../solver/electromagnetic/pml.h"
namespace simpla
{
template<typename TM>
struct ExplicitEMContext: public BaseContext
{
public:
	typedef BaseContext base_type;
	typedef TM mesh_type;
	typedef LuaObject configure_type;

	DEFINE_FIELDS (TM)
public:
	typedef ExplicitEMContext<TM> this_type;

	ExplicitEMContext();
	~ExplicitEMContext();
	void Deserialize(configure_type const & cfg);
	void Serialize(configure_type * cfg) const;
	std::ostream & Serialize(std::ostream & os) const;
	void NextTimeStep(double dt);
	void DumpData() const;

	inline ParticleCollection<mesh_type> & GetParticleCollection()
	{
		return particle_collection_;
	}
public:

	mesh_type mesh;

	typedef MediaTag<mesh_type> mediatag_type;
	typedef typename mediatag_type::tag_type tag_type;

	mediatag_type media_tag;

	Form<1> E, dE;
	Form<1> Jext;
	Form<2> B, dB;
	RVectorForm<0> B0;
	RForm<0> n0;

	ColdFluidEM<mesh_type> cold_fluid_;
	PML<mesh_type> pml_;
	ParticleCollection<mesh_type> particle_collection_;

	typedef typename ParticleCollection<mesh_type>::particle_type particle_type;

	bool isCompactStored_;

	typedef LuaObject field_function;

	FieldFunction<decltype(Jext), field_function> j_src_;

	std::map<std::string, std::function<void()> > boundary_condition_;
}
;

template<typename TM>
ExplicitEMContext<TM>::ExplicitEMContext() :
		E(mesh), dE(mesh), B(mesh), dB(mesh), Jext(mesh), B0(mesh), n0(mesh),

		cold_fluid_(mesh),

		pml_(mesh),

		particle_collection_(mesh),

		isCompactStored_(true),

		media_tag(mesh)

{
	particle_collection_.template RegisterFactory<PICEngineFull<mesh_type> >();
//	particle_collection_.template RegisterFactory<PICEngineDeltaF<mesh_type> >();
//
//	particle_collection_.template RegisterFactory<GGauge<mesh_type, 0>>("GuidingCenter");
//	particle_collection_.template RegisterFactory<GGauge<mesh_type, 8>>("GGauge8");
//	particle_collection_.template RegisterFactory<GGauge<mesh_type, 32>>("GGauge32");
}

template<typename TM>
ExplicitEMContext<TM>::~ExplicitEMContext()
{
}
template<typename TM>
void ExplicitEMContext<TM>::Deserialize(LuaObject const & cfg)
{
	base_type::description = cfg["Description"].as<std::string>();

	mesh.Deserialize(cfg["Grid"]);

	cold_fluid_.Deserialize(cfg["FieldSolver"]["ColdFluid"]);

	pml_.Deserialize(cfg["FieldSolver"]["PML"]);

	particle_collection_.Deserialize(cfg["Particles"]);

	auto init_value = cfg["InitValue"];

	auto gfile = cfg["GFile"];

	dE.Init();
	dB.Init();
	if (gfile.empty())
	{
		LoadField(init_value["n0"], &n0);
		LoadField(init_value["B0"], &B0);
		LoadField(init_value["E"], &E);
		LoadField(init_value["B"], &B);
		LoadField(init_value["J"], &Jext);

		LOGGER << "Load Initial Fields." << DONE;
	}
	else
	{
		UNIMPLEMENT << "TODO: use g-file initialize field, set boundary condition!";
	}

	{
		LuaObject jSrcCfg = cfg["CurrentSrc"];

		if (!jSrcCfg.empty())
		{
			j_src_.SetFunction(jSrcCfg["Fun"]);

			j_src_.SetDefineDomain(mesh, jSrcCfg["Points"].as<std::vector<coordinates_type>>());

			LOGGER << "Load Current Source ." << DONE;
		}
	}

	{

		media_tag.Deserialize(cfg["Media"]);

		if (!media_tag.empty())
		{
			LuaObject boundary = cfg["Boundary"];

			for (auto const & obj : boundary)
			{
				std::string type = "";

				obj.second["Type"].as<std::string>(&type);

				tag_type in = media_tag.GetTagFromString(obj.second["In"].as<std::string>());
				tag_type out = media_tag.GetTagFromString(obj.second["Out"].as<std::string>());

				if (type == "PEC")
				{
					boundary_condition_.emplace(

					"PEC",

					[in,out,this]()
					{

						media_tag.template SelectBoundaryCell<1>(
								[this](index_type const &s)
								{
									(this->E)[s]=0;
								}
								,in,out,mediatag_type::ON_BOUNDARY );

					}

					);
				}
				else
				{
					UNIMPLEMENT << "Unknown boundary type [" << type << "]";
				}

				LOGGER << "Load Boundary " << type << DONE;

			}
		}
	}

	LOGGER << ">>>>>>> Initialization  Complete! <<<<<<<< ";

}

template<typename TM>
void ExplicitEMContext<TM>::Serialize(configure_type * cfg) const
{
}
template<typename TM>
std::ostream & ExplicitEMContext<TM>::Serialize(std::ostream & os) const
{

	os << "Description=\"" << base_type::description << "\" \n";

	os << mesh << "\n"

	<< media_tag << "\n"

	<< " FieldSolver={ \n"

	<< cold_fluid_ << ",\n"

	<< pml_ << ",\n" << "} \n"

	<< particle_collection_ << "\n"

	;

	os << "Function={";
	for (auto const & p : boundary_condition_)
	{
		os << "\"" << p.first << "\",\n";
	}
	os << "}\n";

	GLOBAL_DATA_STREAM.OpenGroup("/InitValue");

	os

	<< "InitValue={" << "\n"

	<< "	n0 = " << DUMP(n0) << ",\n"

	<< "	E = " << DUMP(E) << ",\n"

	<< "	B = " << DUMP(B) << ",\n"

	<< "	J = " << DUMP(Jext) << ",\n"

	<< "	B0 = " << DUMP(B0) << "\n"

	<< "}" << "\n"

	;
	return os;
}
template<typename TM>
void ExplicitEMContext<TM>::NextTimeStep(double dt)
{
	dt = std::isnan(dt) ? mesh.GetDt() : dt;

	base_type::NextTimeStep(dt);

	DEFINE_PHYSICAL_CONST(mesh.constants);

	LOGGER

	<< "Simulation Time = "

	<< (base_type::GetTime() / mesh.constants["s"]) << "[s]"

	<< " dt = " << (dt / mesh.constants["s"]) << "[s]";

	Form<1> Jext(mesh);

	Jext.Fill(0);

	if (dB.empty())
		dB = -Curl(E) * dt;

	//************************************************************
	// Compute Cycle Begin
	//************************************************************

	if (!j_src_.empty())
	{
		LOGGER << "Current Source:" << DUMP(Jext);
		j_src_(&Jext, base_type::GetTime());
	}

	particle_collection_.Sort();
	// B(t=0) E(t=0) particle(t=0) Jext(t=0)
	particle_collection_.Collect(&Jext, E, B);

	// B(t=0 -> 1/2)
	LOG_CMD(B += dB * 0.5);

	if (!pml_.empty())
	{
		pml_.NextTimeStepE(dt, B, &dE);
	}
	else
	{
		LOG_CMD(dE = (Curl(B) / mu0) * (dt / epsilon0));
	}

	LOG_CMD(dE -= Jext * (dt / epsilon0));

	// J(t=1/2-  to 1/2 +)= (E(t=1/2+)-E(t=1/2-))/dt
	if (!cold_fluid_.empty())
	{
		cold_fluid_.NextTimeStep(dt, E, B, &dE);
	}
	// E(t=0  -> 1/2  )
	LOG_CMD(E += dE * 0.5);

	//  particle(t=0 -> 1)
	particle_collection_.NextTimeStep(dt, E, B);

	//  E(t=1/2  -> 1)
	LOG_CMD(E += dE * 0.5);

	//	LOG_CMD(E += (Curl(B / mu0) - Jext - J) * (0.5 * dt / epsilon0));

	if (boundary_condition_.find("PEC") != boundary_condition_.end())
	{
		LOG_CMD(boundary_condition_["PEC"]());
	}
	if (!pml_.empty())
	{
		pml_.NextTimeStepB(dt, E, &dB);
	}
	else
	{
		LOG_CMD(dB = -Curl(E) * dt);
	}

	//  B(t=1/2 -> 1)
	LOG_CMD(B += dB * 0.5);

	//************************************************************
	// Compute Cycle End
	//************************************************************

}
template<typename TM>
void ExplicitEMContext<TM>::DumpData() const
{
	GLOBAL_DATA_STREAM.OpenGroup("/DumpData");

	LOGGER << "Dump E to " << Data(E.data(), "E", E.GetShape(), isCompactStored_);

	LOGGER << "Dump B to " << Data(B.data(), "B", B.GetShape(), isCompactStored_);

	LOGGER << "Dump J to " << Data(Jext.data(), "J", Jext.GetShape(), isCompactStored_);

	cold_fluid_.DumpData();

//	particle_collection_.DumpData();

}
}
	// namespace simpla

#endif /* EXPLICIT_EM_H_ */
