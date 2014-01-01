/*
 * cold_fluid.h
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#ifndef COLD_FLUID_H_
#define COLD_FLUID_H_

#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <memory>

#include "../../../src/fetl/fetl.h"
#include "../../../src/fetl/load_field.h"
#include "../../../src/fetl/save_field.h"
#include "../../../src/utilities/log.h"
#include "../../../src/utilities/lua_state.h"
#include "../../../src/utilities/pretty_stream.h"
#include "../../../src/physics/physical_constants.h"

#include "../../../src/engine/fieldsolver.h"
namespace simpla
{

template<typename TM>
class ColdFluidEM: public FieldSolver<TM>
{
public:
	DEFINE_FIELDS(TM)

	typedef Mesh mesh_type;

	typedef ColdFluidEM<mesh_type> this_type;

	mesh_type const & mesh;
private:

	struct Species
	{
		Real m;
		Real Z;
		RForm<0> n;
		VectorForm<0> J;

		Species(Real pm, Real pZ, mesh_type const &mesh) :
				m(pm), Z(pZ), n(mesh), J(mesh)
		{
		}
		~Species()
		{
		}

	};
	std::map<std::string, std::shared_ptr<Species>> sp_list_;
	VectorForm<0> Ev;
public:

	ColdFluidEM(mesh_type const & pmesh) :
			mesh(pmesh), Ev(pmesh)
	{
	}

	~ColdFluidEM()
	{
	}

	inline bool empty() override // STL style
	{
		return sp_list_.empty();
	}
	void Deserialize(LuaObject const&cfg) override;
	std::ostream & Serialize(std::ostream & os) const override;

	void DumpData() const;

private:
	template<typename TE, typename TB> inline
	void _NextTimeStepE(Real dt, TE const &dE, TB const &B0, TE *E);

}
;
template<typename TM>
template<typename TE, typename TB> inline
void ColdFluidEM<TM>::_NextTimeStepE(Real dt, TE const &E, TB const &B, TE *dE)
{
	if (sp_list_.empty())
		return;

	DEFINE_PHYSICAL_CONST(mesh.constants());

	LOGGER << "Push Cold Fluid.";

	VectorForm<0> cB0(mesh);
	MapTo(B, &cB0);

	RVectorForm<0> B0(mesh);
	B0 = real(cB0);

	RForm<0> BB(mesh);

	BB = Dot(B0, B0);

	if (Ev.empty())
		MapTo(E, &Ev);

	RForm<0> a(mesh);
	RForm<0> b(mesh);
	RForm<0> c(mesh);

	a.Fill(0.0);
	b.Fill(0);
	c.Fill(0);

	VectorForm<0> dEv(mesh);

	MapTo(*dE, &dEv);

	Ev += dEv * 0.5;

	VectorForm<0> Q(mesh);

	Q = 0;

	VectorForm<0> K(mesh);

	for (auto &v : sp_list_)
	{

		auto & ns = v.second->n;
		auto & Js = v.second->J;
		Real ms = v.second->m * proton_mass;
		Real Zs = v.second->Z * elementary_charge;

		Real as = (dt * Zs) / (2.0 * ms);

		a += ns * Zs / (BB * as * as + 1);

		b += ns * Zs * as / (BB * as * as + 1);

		c += ns * Zs * as * as / (BB * as * as + 1);

		Q -= Js;

		K = Js + Cross(Js, B0) * as + Ev * ns * Zs * as;

		Js = (K + Cross(K, B0) * as + Dot(K, B0) * B0 * as * as) / (BB * as * as + 1);

		Q -= Js;

	}

	Q *= (0.5 * dt / epsilon0);
	Q += Ev;

	a *= (0.5 * dt) / epsilon0;
	b *= (0.5 * dt) / epsilon0;
	c *= (0.5 * dt) / epsilon0;
	a += 1;

	Ev = (Q * a - Cross(Q, B0) * b + Dot(Q, B0) * B0 * ((b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);

	for (auto &v : sp_list_)
	{
		auto & ns = v.second->n;
		auto & Js = v.second->J;
		auto ms = v.second->m * proton_mass;
		auto Zs = v.second->Z * elementary_charge;

		Real as = (dt * Zs) / (2.0 * ms);

		Js += (Ev + Cross(Ev, B0) * as + Dot(Ev, B0) * B0 * (as * as)) * ((as * Zs * ns) / (BB * as * as + 1));
	}

	Ev += dEv * (0.5);

	MapTo(Ev, dE);

	*dE -= E;

}

template<typename TM>
inline void ColdFluidEM<TM>::Deserialize(LuaObject const&cfg)
{
	if (cfg.empty())
		return;

	for (auto const & p : cfg)
	{

		std::string key;

		if (!p.first.is_number())
		{
			key = p.first.as<std::string>();
		}
		else
		{
			p.second.GetValue("Name", &key);
		}

		std::shared_ptr<Species> sp(
				new Species(p.second["m"].template as<Real>(1.0), p.second["Z"].template as<Real>(1.0), mesh));

		sp->n.Init();
		sp->J.Init();

		if (!LoadField(p.second["n"], &(sp->n)))
			sp->n.Fill(0);

		if (!LoadField(p.second["J"], &(sp->J)))
			sp->J.Fill(0);

		sp_list_.emplace(std::make_pair(key, sp));

	}

	LOGGER << "Load Cold Fluid solver" << DONE;

}

template<typename TM>
void ColdFluidEM<TM>::DumpData() const
{
	GLOBAL_DATA_STREAM.OpenGroup("/DumpData");

	for (auto const & p : sp_list_)
	{
		LOGGER << "Dump " << "n_" + p.first << " to "
		<< Data(p.second->n.data(), "n_" + p.first, p.second->n.GetShape(), true);

		LOGGER << "Dump " << "J_" + p.first << " to "
		<< Data(p.second->J.data(), "J_" + p.first, p.second->J.GetShape(), true);
	}
}

template<typename TM>
std::ostream & ColdFluidEM<TM>::Serialize(std::ostream & os) const
{
	os << "\tColdFluid = {";

	for (auto const & p : sp_list_)
	{
		os << "\n\t" << p.first

		<< " = { " << " m =" << p.second->m << "," << " Z =" << p.second->Z << ",\n"

		<< "\t n0 = " << Data(p.second->n.data(), "n_" + p.first, p.second->n.GetShape()) << "\n"

		<< "\t J0 = " << Data(p.second->J.data(), "J_" + p.first, p.second->J.GetShape()) << "\n"

		<< "\t},\n";
	}
	os << "}";

	return os;
}

template<typename TM>
inline std::ostream & operator<<(std::ostream & os, ColdFluidEM<TM> const &self)
{
	return self.Serialize(os);
}

}  // namespace simpla

#endif /* COLD_FLUID_H_ */
