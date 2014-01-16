/*
 * explicit_em_impl.h
 *
 *  Created on: 2013年12月29日
 *      Author: salmon
 */

#ifndef EXPLICIT_EM_IMPL_H_
#define EXPLICIT_EM_IMPL_H_

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

#include "../../src/mesh/field_convert.h"

#include "../../src/mesh/media_tag.h"

#include "../../src/fetl/field_function.h"

#include "../../src/utilities/log.h"
#include "../../src/utilities/lua_state.h"
#include "../../src/io/data_stream.h"

#include "../../src/particle/particle.h"
#include "../../src/particle/particle_collection.h"
#include "../../src/particle/pic_engine_full.h"
#include "../../src/particle/pic_engine_deltaf.h"
#include "../../src/particle/pic_engine_ggauge.h"

#include "../../src/engine/fieldsolver.h"

#include "../solver/solver.h"

#include "../../src/utilities/geqdsk.h"

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
	void DumpData(std::string const & path = "") const;

	inline ParticleCollection<mesh_type> & GetParticleCollection()
	{
		return particle_collection_;
	}
public:

	mesh_type mesh;

	bool isCompactStored_;

	Form<1> E;
	Form<2> B;
	Form<1> J;

	std::shared_ptr<FieldSolver<mesh_type> > cold_fluid_;
	std::shared_ptr<FieldSolver<mesh_type> > pml_;

	std::shared_ptr<ParticleCollection<mesh_type>> particle_collection_;
	typedef typename ParticleCollection<mesh_type>::particle_type particle_type;

	typedef LuaObject field_function;

	FieldFunction<Form<1>, field_function> j_src_;

	std::multimap<std::string, std::function<void(Real dt)> > field_boundary_;
	std::multimap<std::string, std::function<void(ParticleBase<mesh_type>*, Real dt)> > particle_boundary_;
}
;

template<typename TM>
ExplicitEMContext<TM>::ExplicitEMContext()
		: isCompactStored_(true), E(mesh), B(mesh), J(mesh),

		cold_fluid_(nullptr), pml_(nullptr), particle_collection_(nullptr)
{
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

	if (!cfg["GFile"].empty())
	{
		typedef TM mesh_type;

		base_type::description = cfg["Description"].as<std::string>();

		GEqdsk geqdsk(cfg["GFile"].as<std::string>());

		mesh.SetExtent(geqdsk.GetMin(), geqdsk.GetMax());

		mesh.SetDimension(geqdsk.GetDimension());

		mesh.Update();

		RForm<EDGE> B1(mesh);

		B1.Fill(0);

		mesh.SerialTraversal(EDGE,

		[&](typename mesh_type::index_type s,typename mesh_type::coordinates_type const &x)
		{
			B1[s] = mesh.template GetWeightOnElement<FACE>(geqdsk.B(x),s);
		});

		MapTo(B1, &B);
	}
	else if (!cfg["InitValue"].empty())
	{
		auto init_value = cfg["InitValue"];

		LOGGER << "Load E";
		LoadField(init_value["E"], &E);
		LOGGER << "Load B";
		LoadField(init_value["B"], &B);
		LOGGER << "Load J";
		LoadField(init_value["J"], &J);

		LOGGER << "Load Initial Fields." << DONE;
	}

	{
		if (!cfg["FieldSolver"]["ColdFluid"].empty())
		{
			cold_fluid_ = CreateSolver(mesh, "ColdFluid");
			cold_fluid_->Deserialize(cfg["FieldSolver"]["ColdFluid"]);
		}
		if (!cfg["FieldSolver"]["PML"].empty())
		{
			pml_ = CreateSolver(mesh, "PML");
			pml_->Deserialize(cfg["FieldSolver"]["PML"]);
		}

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
		LuaObject boundary = cfg["Boundary"];
		if (!(mesh.tags().empty() || boundary.empty()))
		{

			for (auto const & item : boundary)
			{
				auto object = item.second["Object"].template as<std::string>();
				auto type = item.second["Type"].template as<std::string>();
				auto function = item.second["Function"].template as<std::string>();

				tag_type in = mesh.tags().GetTagFromString(item.second["In"].as<std::string>());
				tag_type out = mesh.tags().GetTagFromString(item.second["Out"].as<std::string>());

				if (function == "PEC")
				{
					field_boundary_.emplace(object,

					[in,out,this](Real dt)
					{
						auto selector=mesh.tags().template BoundarySelector<EDGE>(in,out);

						this->mesh.ParallelTraversal(EDGE,
								[this,selector](index_type s)
								{
									if(selector(s) )(this->E)[s]=0;
								}
						);

					}

					);
				}
				else if (type == "Particle" && function == "Reflect")
				{
					particle_boundary_.emplace(object,

					[in,out,this](ParticleBase<mesh_type> * p,Real dt)
					{
						p->Boundary(ParticleBase<mesh_type>::REFELECT,in,out,dt,this->E,this->B);
					}

					);
				}
				else if (type == "Particle" && function == "Absorb")
				{
					particle_boundary_.emplace(object,

					[in,out,this](ParticleBase<mesh_type>* p,Real dt)
					{
						p->Boundary(ParticleBase<mesh_type>::ABSORB,in,out,dt,this->E,this->B);
					}

					);
				}
				else
				{
					UNIMPLEMENT << "Unknown boundary type!" << " [function = " << function << " type= " << type
					        << " object =" << object << " ]";
				}

				LOGGER << "Load Boundary " << type << DONE;

			}
		}

	}

	{
		LuaObject particles = cfg["Particles"];
		if (!particles.empty())
		{
			particle_collection_ = std::shared_ptr<ParticleCollection<mesh_type> >(
			        new ParticleCollection<mesh_type>(mesh));

			particle_collection_->template RegisterFactory<PICEngineFull<mesh_type> >();
			particle_collection_->template RegisterFactory<PICEngineDeltaF<mesh_type> >();
			particle_collection_->template RegisterFactory<PICEngineGGauge<mesh_type, 8>>("GGauge8");
			particle_collection_->template RegisterFactory<PICEngineGGauge<mesh_type, 32>>("GGauge32");

			particle_collection_->Deserialize(particles);
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

	os << "Grid = " << mesh << "\n"

	<< " FieldSolver={ \n";

	if (cold_fluid_ != nullptr)
		os << *cold_fluid_ << ",\n";

	if (pml_ != nullptr)
		os << *pml_ << ",\n";

	os << "} \n";

	if (particle_collection_ != nullptr)
		os << *particle_collection_ << "\n";

	os << "Function={";
	for (auto const & p : field_boundary_)
	{
		os << "\"" << p.first << "\",\n";
	}
	os << "}\n"

	<< "Fields={" << "\n"

	<< "	E = " << DUMP(E) << ",\n"

	<< "	B = " << DUMP(B) << ",\n"

	<< "	J = " << DUMP(J) << ",\n"

	<< "}" << "\n"

	;
	return os;
}
template<typename TM>
void ExplicitEMContext<TM>::NextTimeStep(double dt)
{
	dt = std::isnan(dt) ? mesh.GetDt() : dt;

	if (!mesh.CheckCourant(dt))
		VERBOSE << "dx/dt > c, Courant condition is violated! ";

	base_type::NextTimeStep(dt);

	DEFINE_PHYSICAL_CONST(mesh.constants());

	LOGGER

	<< "Simulation Time = "

	<< (base_type::GetTime() / mesh.constants()["s"]) << "[s]"

	<< " dt = " << (dt / mesh.constants()["s"]) << "[s]";

	//************************************************************
	// Compute Cycle Begin
	//************************************************************

	Form<1> dE(mesh);
	dE.Fill(0);

	if (pml_ != nullptr)
	{
		pml_->NextTimeStepE(dt, E, B, &dE);
	}
	else
	{
		LOG_CMD(dE += Curl(B) / (mu0 * epsilon0));
	}

	LOG_CMD(dE -= J / epsilon0);

	if (!j_src_.empty())
	{

		J.Fill(0);
		j_src_(&J, base_type::GetTime());
		LOGGER << "Current Source:" << DONE;
		LOG_CMD(dE -= J / epsilon0);

	}

	// J(t=1/2-  to 1/2 +)= (E(t=1/2+)-E(t=1/2-))/dt
	if (cold_fluid_ != nullptr)
	{
		cold_fluid_->NextTimeStepE(dt, E, B, &dE);
	}

	// E(t=0  -> 1/2  )
	LOG_CMD(E += dE * 0.5 * dt);

	{
		int count = 0;
		auto range = field_boundary_.equal_range("E");
		for (auto fun_it = range.first; fun_it != range.second; ++fun_it)
		{
			fun_it->second(dt);
			++count;
		}
		if (count > 0)
			LOGGER << "Apply [" << count << "] boundary conditions on E " << DONE;
	}

	if (particle_collection_ != nullptr)
		particle_collection_->NextTimeStep(dt, E, B);	// particle(t=0 -> 1)

	{
		for (auto & p : *particle_collection_)
			for (auto & fun : particle_boundary_)
			{
				if (fun.first == "" || p.first == fun.first)
				{
					fun.second(p.second.get(), dt);

					LOGGER << "Apply boundary conditions on Particle [" << p.first << "] " << DONE;
				}

			}

	}
	//  E(t=1/2  -> 1)
	LOG_CMD(E += dE * 0.5 * dt);

	{
		int count = 0;
		auto range = field_boundary_.equal_range("E");
		for (auto fun_it = range.first; fun_it != range.second; ++fun_it)
		{
			fun_it->second(dt);
			++count;
		}
		if (count > 0)
			LOGGER << "Apply [" << count << "] boundary conditions on E " << DONE;
	}

	Form<2> dB(mesh);

	dB.Fill(0);

	if (pml_ != nullptr)
	{
		pml_->NextTimeStepB(dt, E, B, &dB);
	}
	else
	{
		LOG_CMD(dB += -Curl(E));
	}

	//  B(t=1/2 -> 1)
	LOG_CMD(B += dB * 0.5 * dt);
	{
		int count = 0;
		auto range = field_boundary_.equal_range("B");
		for (auto fun_it = range.first; fun_it != range.second; ++fun_it)
		{
			fun_it->second(dt);
			++count;
		}
		if (count > 0)
			LOGGER << "Apply [" << count << "] boundary conditions on B " << DONE;
	}

	if (particle_collection_ != nullptr)
	{
		particle_collection_->Sort();
		J.Fill(0);
		// B(t=0) E(t=0) particle(t=0) Jext(t=0)
		particle_collection_->Collect(&J, E, B);
	}

	// B(t=0 -> 1/2)
	LOG_CMD(B += dB * 0.5 * dt);

	{
		int count = 0;
		auto range = field_boundary_.equal_range("B");
		for (auto fun_it = range.first; fun_it != range.second; ++fun_it)
		{
			fun_it->second(dt);
			++count;
		}
		if (count > 0)
			LOGGER << "Apply [" << count << "] boundary conditions on B " << DONE;
	}

	//************************************************************
	// Compute Cycle End
	//************************************************************

}
template<typename TM>
void ExplicitEMContext<TM>::DumpData(std::string const & path) const
{
	GLOBAL_DATA_STREAM.OpenGroup(path);

	LOGGER << DUMP(E);
	LOGGER << DUMP(B);
	LOGGER << DUMP(J);

	if(cold_fluid_!=nullptr) cold_fluid_->DumpData(path);
	if(pml_!=nullptr) pml_->DumpData(path);

}
}
// namespace simpla

#endif /* EXPLICIT_EM_IMPL_H_ */
