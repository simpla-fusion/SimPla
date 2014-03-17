/*
 * explicit_em_impl.h
 *
 *  Created on: 2013年12月29日
 *      Author: salmon
 */

#ifndef EXPLICIT_EM_IMPL_H_
#define EXPLICIT_EM_IMPL_H_

#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>

// Misc
#include "../../src/utilities/log.h"

// Field expression
#include "../../src/fetl/field.h"
#include "../../src/fetl/ntuple.h"
#include "../../src/fetl/primitives.h"
#include "../../src/mesh/field_convert.h"

// Data IO
#include "../../src/fetl/save_field.h"
#include "../../src/io/data_stream.h"

// Modeling
#include "../../src/modeling/material.h"
#include "../../src/utilities/geqdsk.h"

// Field solver
#include "../../src/modeling/constraint.h"
#include "../solver/electromagnetic/solver.h"

// Particle
#include "../../src/particle/particle_factory.h"

namespace simpla
{
template<typename TM>
struct ExplicitEMContext
{
public:

	typedef TM mesh_type;

	DEFINE_FIELDS (TM)

	typedef ExplicitEMContext<mesh_type> this_type;

	ExplicitEMContext();

	~ExplicitEMContext();

	template<typename TDict> void Load(TDict const & dict);

	void Save(std::ostream & os) const;

	void NextTimeStep();

	void DumpData(std::string const & path = "") const;

public:

	mesh_type mesh;

	std::string description;

	Material<mesh_type> tags_;
	bool isCompactStored_;

	Form<EDGE> E, dE;
	Form<FACE> B, dB;
	Form<VERTEX> phi; // electrostatic potential

	Form<EDGE> J;     // current density
	Form<EDGE> J0;     //background current density J0+Curl(B(t=0))=0
	Form<VERTEX> rho; // charge density

	typedef decltype(E) TE;
	typedef decltype(B) TB;
	typedef decltype(J) TJ;

	std::function<void(Real, TE const &, TB const &, TE*)> CalculatedE;

	std::function<void(Real, TE const &, TB const &, TB*)> CalculatedB;

	void ApplyConstraintToE(TE* pE)
	{
		if (constraintToE_.size() > 0)
		{
			LOGGER << "Apply Constraint to E";
			for (auto const & foo : constraintToE_)
			{
				foo(pE);
			}
		}
	}
	void ApplyConstraintToB(TB* pB)
	{
		if (constraintToB_.size() > 0)
		{
			LOGGER << "Apply Constraint to B";
			for (auto const & foo : constraintToB_)
			{
				foo(pB);
			}
		}
	}
	void ApplyConstraintToJ(TJ* pJ)
	{
		if (constraintToJ_.size() > 0)
		{
			LOGGER << "Apply Constraint to J";
			for (auto const & foo : constraintToJ_)
			{
				foo(pJ);
			}
		}
	}

private:

	std::list<std::function<void(TE*)> > constraintToE_;

	std::list<std::function<void(TB*)> > constraintToB_;

	std::list<std::function<void(TJ*)> > constraintToJ_;

	typedef ParticleWrap<TE, TB, TJ> ParticleType;

	std::map<std::string, ParticleType> particles_;

}
;

template<typename TM>
ExplicitEMContext<TM>::ExplicitEMContext()
		: isCompactStored_(true), tags_(mesh),

		E(mesh), B(mesh), J(mesh), J0(mesh), dE(mesh), dB(mesh), rho(mesh), phi(mesh)
{
	DEFINE_PHYSICAL_CONST(mesh.constants());
	Real ic2 = 1.0 / (mu0 * epsilon0);
	CalculatedE = [ic2](Real dt, TE const & , TB const & pB, TE* pdE)
	{	LOG_CMD(*pdE += Curl(pB)*ic2 *dt);};

	CalculatedB = [](Real dt, TE const & pE, TB const &, TB* pdB)
	{	LOG_CMD(*pdB -= Curl(pE)*dt);};

}

template<typename TM>
ExplicitEMContext<TM>::~ExplicitEMContext()
{
}
template<typename TM> template<typename TDict>
void ExplicitEMContext<TM>::Load(TDict const & dict)
{
	description = dict["Description"].template as<std::string>();

	mesh.Load(dict["Grid"]);

	B.Clear();
	J.Clear();
	J0.Clear();
	E.Clear();

	dB.Clear();
	dE.Clear();

	if (dict["GFile"])
	{

		GEqdsk geqdsk(dict["GFile"].template as<std::string>());

		nTuple<3, Real> xmin, xmax;

		xmin[0] = geqdsk.GetMin()[0];
		xmin[1] = geqdsk.GetMin()[1];
		xmin[2] = 0;
		xmax[0] = geqdsk.GetMax()[0];
		xmax[1] = geqdsk.GetMax()[1];
		xmax[2] = 0;

		mesh.SetExtent(xmin, xmax);

		mesh.Update();

		tags_.Update();

		mesh.template Traversal<FACE>(

		[&](typename mesh_type::index_type s )
		{
			auto x= mesh.GetCoordinates(s);
			B[s] = mesh.template Sample<FACE>(Int2Type<FACE>(),s,geqdsk.B(x));
		});

		J0 = Curl(B) / mesh.constants()["permeability of free space"];

		tags_.Add("Plasma", geqdsk.Boundary());
		tags_.Add("Vacuum", geqdsk.Limiter());

		LOGGER << "Load GFile" << DONE;
	}

	J = J0;

	if (dict["InitValue"])
	{
		auto init_value = dict["InitValue"];

		LoadField(init_value["E"], &E);
		LOGGER << "Load E" << DONE;
		LoadField(init_value["B"], &B);
		LOGGER << "Load B" << DONE;
		LoadField(init_value["J"], &J);
		LOGGER << "Load J" << DONE;
	}

	LOGGER << "Load Particles";
	for (auto const &opt : dict["Particles"])
	{
		ParticleWrap<TE, TB, TJ> p;

		if (CreateParticle<Mesh, TE, TB, TJ>(mesh, opt.second, &p))
			particles_.emplace(std::make_pair(opt.first.template as<std::string>(), p));
	}

	LOGGER << "Load Constraints";
	for (auto const & item : dict["Constraints"])
	{
		auto dof = item.second["DOF"].template as<std::string>();

		if (dof == "E")
		{
			constraintToE_.push_back(CreateConstraint<TE>(tags_, item.second));
		}
		else if (dof == "B")
		{
			constraintToB_.push_back(CreateConstraint<TB>(tags_, item.second));
		}
		else if (dof == "J")
		{
			constraintToJ_.push_back(CreateConstraint<TJ>(tags_, item.second));
		}
		else
		{
			//TODO Add particles constraints
			UNIMPLEMENT2("Unknown Constraints!!");
			continue;
		}

		LOGGER << "Add constraint to " << dof << DONE;
	}

	CreateEMSolver(dict["FieldSolver"], mesh, &CalculatedE, &CalculatedB);

	LOGGER << "Load ExplicitEMContext " << DONE;

}

template<typename TM>
void ExplicitEMContext<TM>::Save(std::ostream & os) const
{

	os << "Description=\"" << description << "\" \n";

	os << "Grid = { \n" << mesh << "\n";

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
	os << "\n}\n"

	<< "Fields={" << "\n"

	<< "	E = " << Dump(E, "E", false) << ",\n"

	<< "	B = " << Dump(B, "B", false) << ",\n"

	<< "	J = " << Dump(J, "J", false) << ",\n"

	<< "}" << "\n"

	;
}
template<typename TM>
void ExplicitEMContext<TM>::NextTimeStep()
{
	Real dt = mesh.GetDt();

	mesh.NextTimeStep();

	DEFINE_PHYSICAL_CONST(mesh.constants());

	VERBOSE

	<< "Simulation Time = "

	<< (mesh.GetTime() / mesh.constants()["s"]) << "[s]"

	<< " dt = " << (dt / mesh.constants()["s"]) << "[s]";

	//************************************************************
	// Compute Cycle Begin
	//************************************************************

	LOG_CMD(dE = -J / epsilon0 * dt);

	// dE = Curl(B)*dt
	CalculatedE(dt, E, B, &dE);

	// E(t=0  -> 1/2  )
	LOG_CMD(E += dE * 0.5);

	ApplyConstraintToE(&E);

	for (auto &p : particles_)
	{
		p.second.NextTimeStep(dt, E, B);	// particle(t=0 -> 1)
	}

	//  E(t=1/2  -> 1)
	LOG_CMD(E += dE * 0.5);

	ApplyConstraintToE(&E);

	dB.Clear();

	CalculatedB(dt, E, B, &dB);

	//  B(t=1/2 -> 1)
	LOG_CMD(B += dB * 0.5);

	ApplyConstraintToB(&B);

	J = J0;

	ApplyConstraintToJ(&J);

	for (auto &p : particles_)
	{
		// B(t=0) E(t=0) particle(t=0) Jext(t=0)
		p.second.Scatter(&J, E, B);
	}

	// B(t=0 -> 1/2)
	LOG_CMD(B += dB * 0.5);

	ApplyConstraintToB(&B);

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
