/*
 * CoRectMesh.h
 *
 *  Created on: 2013年12月12日
 *      Author: salmon
 */

#ifndef RECT_MESH_H_
#define RECT_MESH_H_

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
//#include "../../src/fetl/fetl.h"
#include "../../src/fetl/field_function.h"
#include "../../src/fetl/field.h"
#include "../../src/fetl/ntuple.h"
#include "../../src/fetl/primitives.h"
#include "../../src/io/data_stream.h"
#include "../../src/mesh/media_tag.h"
#include "../../src/particle/particle.h"
#include "../../src/particle/pic_engine_default.h"
#include "../../src/particle/pic_engine_deltaf.h"
#include "../../src/utilities/log.h"
#include "../../src/utilities/lua_state.h"
#include "../../src/utilities/singleton_holder.h"
#include "../pic/pic_engine_ggauge.h"
#include "../solver/electromagnetic/cold_fluid.h"

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

	DEFINE_FIELDS (TM)
public:
	typedef Context<TM> this_type;

	Context();
	~Context();
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
	typedef typename mesh_type::scalar_type scalar_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef MediaTag<mesh_type> mediatag_type;
	typedef typename mediatag_type::tag_type tag_type;

	mediatag_type media_tag;

	Form<1> E;
	Form<1> J;
	Form<2> B;
	RVectorForm<0> B0;
	RForm<0> n0;

	ColdFluidEM<mesh_type> cold_fluid_;
	ParticleCollection<mesh_type> particle_collection_;

	typedef typename ParticleCollection<mesh_type>::particle_type particle_type;

	bool isCompactStored_;

//	typedef typename Form<1>::field_value_type field_value_type;
//	typedef std::function<field_value_type(Real, Real, Real, Real)> field_function;
	typedef LuaObject field_function;
	FieldFunction<decltype(J), field_function> j_src_;

	std::map<std::string, std::function<void()> > function_;
}
;

template<typename TM>
Context<TM>::Context()
		: E(mesh), B(mesh), J(mesh), B0(mesh), n0(mesh),

		cold_fluid_(mesh), particle_collection_(mesh), isCompactStored_(true),

		media_tag(mesh)

{

	particle_collection_.template RegisterFactory<GGauge<mesh_type, 0>>("GuidingCenter");
	particle_collection_.template RegisterFactory<GGauge<mesh_type, 8>>("GGauge8");
	particle_collection_.template RegisterFactory<GGauge<mesh_type, 32>>("GGauge32");
	particle_collection_.template RegisterFactory<PICEngineDefault<mesh_type> >("Default");
	particle_collection_.template RegisterFactory<PICEngineDeltaF<mesh_type> >("DeltaF");
}

template<typename TM>
Context<TM>::~Context()
{
}
template<typename TM>
void Context<TM>::Deserialize(LuaObject const & cfg)
{
	base_type::description = cfg["Description"].as<std::string>();

	mesh.Deserialize(cfg["Grid"]);

	cold_fluid_.Deserialize(cfg["FieldSolver"]["ColdFluid"]);

//	particle_collection_.Deserialize(cfg["Particles"]);

	auto init_value = cfg["InitValue"];

	auto gfile = cfg["GFile"];

	if (gfile.empty())
	{
		LoadField(init_value["n0"], &n0);
		LoadField(init_value["B0"], &B0);
		LoadField(init_value["E"], &E);
		LoadField(init_value["B"], &B);
		LoadField(init_value["J"], &J);

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
					function_.emplace(

					"PEC",

					[in,out,this]()
					{

						media_tag.template SelectBoundaryCell<1>(
								[this](index_type const &s)
								{
									(this->E)[s]=0;
								}
								,in,out,mediatag_type::ON_BOUNDARY,mesh_type::DO_PARALLEL );

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
void Context<TM>::Serialize(configure_type * cfg) const
{
}
template<typename TM>
std::ostream & Context<TM>::Serialize(std::ostream & os) const
{

	os << "Description=\"" << base_type::description << "\" \n";

	os << mesh << "\n"

	<< media_tag << "\n"

	<< " FieldSolver={ \n"

	<< cold_fluid_ << "\n"

	<< "} \n";

//	os << particle_collection_ << "\n"

	;

	os << "Function={";
	for (auto const & p : function_)
	{
		os << "\"" << p.first << "\",\n";
	}
	os << "}\n";

	GLOBAL_DATA_STREAM.OpenGroup("/InitValue");

	os

	<< "InitValue={" << "\n"

	<< "	n0 = " << Data(n0.data(), "n0", n0.GetShape()) << ",\n"

	<< "	E = " << Data(E.data(), "E", E.GetShape()) << ",\n"

	<< "	B = " << Data(B.data(), "B", B.GetShape()) << ",\n"

	<< "	J = " << Data(J.data(), "J", J.GetShape()) << ",\n"

	<< "	B0 = " << Data(B0.data(), "B0", n0.GetShape()) << "\n"

	<< "}" << "\n"

	;
	return os;
}
template<typename TM>
void Context<TM>::NextTimeStep(double dt)
{
	dt = std::isnan(dt) ? mesh.GetDt() : dt;

	base_type::NextTimeStep(dt);

	LOGGER

	<< " SimTime = "

	<< (base_type::GetTime() / mesh.constants["s"]) << "[s]"

	<< " dt = " << (dt / mesh.constants["s"]) << "[s]";

	Form<1> Jext(mesh);

	Jext.Fill(0);

	if (!j_src_.empty())
		j_src_(&Jext, base_type::GetTime());

	const double mu0 = mesh.constants["permeability of free space"];
	const double epsilon0 = mesh.constants["permittivity of free space"];
	const double speed_of_light = mesh.constants["speed of light"];
	const double proton_mass = mesh.constants["proton mass"];
	const double elementary_charge = mesh.constants["elementary charge"];

	// B(t=0) E(t=0) particle(t=0) Jext(t=0)
	//	particle_collection_.CollectAll(dt, &Jext, E, B);

	// B(t=0 -> 1/2)
	LOG_CMD(B -= Curl(E) * (0.5 * dt));

	// E(t=0 -> 1/2-)
	LOG_CMD(E += (Curl(B / mu0) - Jext) / epsilon0 * (0.5 * dt));

	// J(t=1/2-  to 1/2 +)= (E(t=1/2+)-E(t=1/2-))/dts
	if (!cold_fluid_.IsEmpty())
	{
		LOG_CMD(cold_fluid_.NextTimeStep(dt, E, B, &J));
	}

	// E(t=1/2-  -> 1/2 +)
	LOG_CMD(E += (-J) / epsilon0 * (0.5 * dt));

	//  particle(t=0 -> 1)
	//	particle_collection_.Push(dt, E, B);

	//  E(t=1/2+ -> 1)
	LOG_CMD(E += (Curl(B / mu0) - Jext - J) / epsilon0 * (0.5 * dt));

	if (function_.find("PEC") != function_.end())
	{
		LOG_CMD(function_["PEC"]());
	}
	//  B(t=1/2 -> 1)
	LOG_CMD(B -= Curl(E) * (0.5 * dt));

}
template<typename TM>
void Context<TM>::DumpData() const
{
	GLOBAL_DATA_STREAM.OpenGroup("/DumpData");

	LOGGER << "Dump E to " << Data(E.data(), "E", E.GetShape(), isCompactStored_);

	LOGGER << "Dump B to " << Data(B.data(), "B", B.GetShape(), isCompactStored_);

	LOGGER << "Dump J to " << Data(J.data(), "J", J.GetShape(), isCompactStored_);

	cold_fluid_.DumpData();

//	particle_collection_.DumpData();

}
}
	// namespace simpla

#endif /* RECT_MESH_H_ */
