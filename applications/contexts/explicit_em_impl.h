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



#include "../solver/solver.h"

#include "../../src/utilities/geqdsk.h"

namespace simpla
{

template<typename ...Args>
void NullFunction(Args const & ...)
{
}

template<typename TM>
struct ExplicitEMContext: public BaseContext
{
public:
	typedef BaseContext base_type;
	typedef TM mesh_type;
	typedef LuaObject configure_type;

	DEFINE_FIELDS (TM)

	Real time_;
public:
	typedef ExplicitEMContext<TM> this_type;

	ExplicitEMContext();

	~ExplicitEMContext();

	void Load(configure_type const & cfg);

	std::ostream & Save(std::ostream & os) const;

	void NextTimeStep(double dt);

	void DumpData(std::string const & path = "") const;

	Real GetTime()
	{
		return time_;
	}

public:

	mesh_type mesh;

	std::string description;

	bool isCompactStored_;

	Form<1> E, dE;
	Form<2> B, dB;
	Form<1> J;

	typedef decltype(E) TE;
	typedef decltype(B) TB;
	typedef decltype(J) TJ;

	std::function<void(Real, TE const &, TB const &, TE*)> CalculatedE;

	std::function<void(Real, TE const &, TB const &, TB*)> CalculatedB;

	std::function<void(TE*)> ApplyBoundaryConditionToE;

	std::function<void(TB*)> ApplyBoundaryConditionToB；

	std::function<void(TJ*, Real)> ApplyCurrentSrcToJ;

	struct ParticleWrap
	{

		std::function<void(Real dt, TE const & E, TB const & B)> NextTimeStep;

		std::function<void(TJ * J, TE const & E, TB const & B)> Collect;

		std::function<std::ostream(std::ostream &)> Save;

		std::function<void(LuaObject const&)> Load;

		std::function<void()> Initialize;

		std::function<void()> Sort;

		std::function<void()> Boundary;

		std::function<void(std::string const &)> DumpData;

	};

	std::map<std::string, ParticleWrap> particles_;

}
;

template<typename TM>
ExplicitEMContext<TM>::ExplicitEMContext()
		: isCompactStored_(true), time_(0), E(mesh), B(mesh), J(mesh), dE(mesh), dB(mesh)
{
}

template<typename TM>
ExplicitEMContext<TM>::~ExplicitEMContext()
{
}
template<typename TM>
void ExplicitEMContext<TM>::Load(LuaObject const & cfg)
{
	description = cfg["Description"].as<std::string>();

	mesh.Deserialize(cfg["Grid"]);

	if (cfg["GFile"])
	{
		typedef TM mesh_type;

		description = cfg["Description"].as<std::string>();

		GEqdsk geqdsk(cfg["GFile"].as<std::string>());

		mesh.SetExtent(geqdsk.GetMin(), geqdsk.GetMax());

		mesh.SetDimension(geqdsk.GetDimension());

		mesh.Update();

		RForm<EDGE> B1(mesh);

		B1.Clear();

		mesh.SerialTraversal(EDGE,

		[&](typename mesh_type::index_type s,typename mesh_type::coordinates_type const &x)
		{
			B1[s] = mesh.template GetWeightOnElement<FACE>(geqdsk.B(x),s);
		});

		MapTo(B1, &B);

		J.Clear();
		E.Clear();

	}
	else if (cfg["InitValue"])
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

	CreateEMFieldSolver(cfg["FieldSolver"], mesh, &CalculatedE, &CalculatedB);

	CreateCurrentSrc(cfg["CurrentSrc"], mesh, &ApplyCurrentSrcToJ);

//	ApplyBoundaryConditionToE=;
//	ApplyBoundaryConditionToB=;

	if (!(mesh.tags() || cfg["Boundary"]))
	{
		LuaObject boundary = cfg["Boundary"];

		for (auto const & item : boundary)
		{
			auto object = item.second["Object"].template as<std::string>();
			auto type = item.second["Type"].template as<std::string>();
			auto function = item.second["Function"].template as<std::string>();

			tag_type in = mesh.tags().GetTagFromString(item.second["In"].as<std::string>());
			tag_type out = mesh.tags().GetTagFromString(item.second["Out"].as<std::string>());

			if (function == "PEC")
			{
				ApplyBoundaryConditionToE =

				[in,out,this](Real dt,TE * pE)
				{
					auto selector=mesh.tags().template BoundarySelector<EDGE>(in,out);

					this->mesh.ParallelTraversal(EDGE,
							[this,selector](index_type s)
							{
								if(selector(s) )(*pE)[s]=0;
							}
					);

				}

				;
			}
//			else if (type == "Particle" && function == "Reflect")
//			{
//				particle_boundary_.emplace(object,
//
//				[in,out,this](ParticleBase<mesh_type> * p,Real dt)
//				{
//					p->Boundary(ParticleBase<mesh_type>::REFELECT,in,out,dt,this->E,this->B);
//				}
//
//				);
//			}
//			else if (type == "Particle" && function == "Absorb")
//			{
//				particle_boundary_.emplace(object,
//
//				[in,out,this](ParticleBase<mesh_type>* p,Real dt)
//				{
//					p->Boundary(ParticleBase<mesh_type>::ABSORB,in,out,dt,this->E,this->B);
//				}
//
//				);
//			}
			else
			{
				UNIMPLEMENT << "Unknown boundary type!" << " [function = " << function << " type= " << type
				        << " object =" << object << " ]";
			}

			LOGGER << "Load Boundary " << type << DONE;

		}

	}
	if (cfg["Particles"])
	{
		LuaObject particles = cfg["Particles"];

		for (auto const &opt : cfg)
		{
			particles_.emplace_bace(CreateParticle<ParticleWrap>(opt))
		}

	}

	LOGGER << ">>>>>>> Initialization  Complete! <<<<<<<< ";

}

template<typename TM>
std::ostream & ExplicitEMContext<TM>::Save(std::ostream & os) const
{

	os << "Description=\"" << description << "\" \n";

	os << "Grid = " << mesh << "\n";

//	os << " FieldSolver={ \n";
//
//	if (cold_fluid_ != nullptr)
//		os << *cold_fluid_ << ",\n";
//
//	if (pml_ != nullptr)
//		os << *pml_ << ",\n";
//
//	os << "} \n";
//
//	if (particles_ != nullptr)
//		os << *particles_ << "\n";
//
//	os << "Function={";
//	for (auto const & p : field_boundary_)
//	{
//		os << "\"" << p.first << "\",\n";
//	}
	os << "}\n"

	<< "Fields={" << "\n"

	<< "	E = " << Dump(E, "E", false) << ",\n"

	<< "	B = " << Dump(B, "B", false) << ",\n"

	<< "	J = " << Dump(J, "J", false) << ",\n"

	<< "}" << "\n"

	;
	return os;
}
template<typename TM>
void ExplicitEMContext<TM>::NextTimeStep(double dt)
{
	dt = std::isnan(dt) ? mesh.GetDt() : dt;

	time_ += dt;

	if (!mesh.CheckCourant(dt))
		VERBOSE << "dx/dt > c, Courant condition is violated! ";

	DEFINE_PHYSICAL_CONST(mesh.constants());

	LOGGER

	<< "Simulation Time = "

	<< (GetTime() / mesh.constants()["s"]) << "[s]"

	<< " dt = " << (dt / mesh.constants()["s"]) << "[s]";

	//************************************************************
	// Compute Cycle Begin
	//************************************************************

	LOG_CMD(ApplyCurrentSrcToJ(&J, GetTime()));

	dE.Clear();

	// dE = Curl(B)*dt
	LOG_CMD(CalculatedE(dt, E, B, &dE))

	LOG_CMD(dE -= J / epsilon0 * dt);

	// E(t=0  -> 1/2  )
	LOG_CMD(E += dE * 0.5);

	LOG_CMD(ApplyBoundaryConditionToE(&E));

	for (auto &p : particles_)
	{
		LOGGER << "Push Particle " << p.first << std::endl;

		p.second.NextTimeStep(dt, E, B);	// particle(t=0 -> 1)

		p.second.Boundary();
	}

	//  E(t=1/2  -> 1)
	LOG_CMD(E += dE * 0.5);

	LOG_CMD(ApplyBoundaryConditionToE(&E));

	Form<2> dB(mesh);

	dB.Clear();

	LOG_CMD(CalculatedB(dt, E, B, &dB));

	//  B(t=1/2 -> 1)
	LOG_CMD(B += dB * 0.5);

	LOG_CMD(ApplyBoundaryConditionToB(&B));

	J.Clear();

	for (auto &p : particles_)
	{
		LOGGER << "Collect Particle " << p.first << std::endl;

		p.Sort();

		// B(t=0) E(t=0) particle(t=0) Jext(t=0)
		p.Collect(&J, E, B);
	}

	// B(t=0 -> 1/2)
	LOG_CMD(B += dB * 0.5);

	LOG_CMD(ApplyBoundaryConditionToB(&B));

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

}
}
// namespace simpla

#endif /* EXPLICIT_EM_IMPL_H_ */
