/*
 * CoRectMesh.h
 *
 *  Created on: 2013年12月12日
 *      Author: salmon
 */

#ifndef RECT_MESH_H_
#define RECT_MESH_H_

#include <iostream>
#include <limits>
#include <string>
#include <cmath>

#include "../../src/engine/basecontext.h"
#include "../../src/fetl/fetl.h"
#include "../../src/io/data_stream.h"
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
	void NextTimeStep(double dt);
	std::ostream & Serialize(std::ostream & os) const;
	void DumpData() const;

	inline ParticleCollection<mesh_type> & GetParticleCollection()
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

	bool dumpInOneDataSet_;
}
;

template<typename TM>
Context<TM>::Context()
		: E1(mesh), B1(mesh), J1(mesh), B0(mesh), n0(mesh),
		        cold_fluid_(mesh), particle_collection_(mesh),
		        dumpInOneDataSet_(true)
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

	cold_fluid_.Deserialize(cfg["Particles"]);

	LOGGER << " Load Cold Fluid [Done]!";

	particle_collection_.Deserialize(cfg["Particles"]);

	LOGGER << " Load Particles [Done]!";

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

	LOGGER << " Load Initial Fields [Done]!";
}

template<typename TM>
void Context<TM>::Serialize(configure_type * cfg) const
        {
}
template<typename TM>
std::ostream & Context<TM>::Serialize(std::ostream & os) const
        {

	os

	<< "Descrition=\"" << base_type::description << "\" \n"

	<< mesh << "\n"

	<< cold_fluid_ << "\n"

	<< particle_collection_ << "\n";

	GLOBAL_DATA_STREAM.OpenGroup("/InitValue");

	os

	<< "InitValue={" << "\n"

	<< "	n0 = " << Data(n0, "n0", n0.GetShape()) << ",\n"

	<< "	E1 = " << Data(E1, "E1", E1.GetShape()) << ",\n"

	<< "	B1 = " << Data(B1, "B1", B1.GetShape()) << ",\n"

	<< "	J1 = " << Data(J1, "J1", J1.GetShape()) << ",\n"

	<< "	B0 = " << Data(B0, "B0", n0.GetShape()) << "\n"

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
}
template<typename TM>
void Context<TM>::DumpData() const
{
	GLOBAL_DATA_STREAM.OpenGroup("/DumpData");

	LOGGER << "Dump E1 to " << Data(E1, "E1", E1.GetShape(), dumpInOneDataSet_);

	LOGGER << "Dump B1 to " << Data(B1, "B1", B1.GetShape(), dumpInOneDataSet_);

	LOGGER << "Dump J1 to " << Data(J1, "J1", J1.GetShape(), dumpInOneDataSet_);

}
}
 // namespace simpla

#endif /* RECT_MESH_H_ */
