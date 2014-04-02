/*
 * cold_fluid.h
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#ifndef COLD_FLUID_H_
#define COLD_FLUID_H_

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
#include "../../../src/mesh/field_convert.h"
namespace simpla
{

template<typename TM>
class ColdFluidEM
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
		Real q;
		RForm<0> n;
		VectorForm<0> J;

		Species(Real pm, Real pZ, mesh_type const &mesh)
				: m(pm), q(pZ), n(mesh), J(mesh)
		{
		}
		~Species()
		{
		}

	};
	std::map<std::string, std::shared_ptr<Species>> sp_list_;
	RVectorForm<0> B0, Ev;
	RForm<0> BB;
	RForm<0> a;
	RForm<0> b;
	RForm<0> c;

	bool enableNonlinear_;
public:

	ColdFluidEM(mesh_type const & pmesh)
			: mesh(pmesh), B0(pmesh), BB(pmesh), enableNonlinear_(false),

			Ev(mesh), a(pmesh), b(pmesh), c(pmesh), dt_(mesh.GetDt())
	{
	}

	~ColdFluidEM()
	{
	}

	inline bool empty() const  // STL style
	{
		return sp_list_.empty();
	}

	template<typename TDict, typename ...Args>
	void Load(TDict const&dict, RForm<0> const & ne, Args const & ...);

	template<typename OS>
	void Print(OS & os) const;

	void DumpData(std::string const & path = "") const;

	template<typename TE, typename TB>
	void NextTimeStepE(Real dt, TE const &E, TB const &B0, TE *dE);

private:

	Real dt_;

}
;
template<typename TM>
template<typename TE, typename TB>
void ColdFluidEM<TM>::NextTimeStepE(Real dt, TE const &E, TB const &B, TE *pdE)
{

	DEFINE_PHYSICAL_CONST(mesh.constants());

	if (sp_list_.empty())
		return;

	LOGGER << "Push E: Cold Fluid. [ Species Number=" << sp_list_.size() << "]";
	VERBOSE << "Nonlinear is " << ((enableNonlinear_) ? "opened" : "closed") << ".";
	TE & dE = *pdE;

	if (BB.empty() || enableNonlinear_)
	{
		B0 = MapTo<VERTEX>(B);
		BB = Dot(B0, B0);
	}
	if (a.empty() || dt_ != dt)
	{
		dt_ = dt;
		a.Clear();
		b.Clear();
		c.Clear();

		for (auto &v : sp_list_)
		{
			auto & ns = v.second->n;
			auto & Js = v.second->J;
			Real ms = v.second->m;
			Real qs = v.second->q;
			Real as = (dt * qs) / (2.0 * ms);

			a += ns * qs * as / (BB * as * as + 1);
			b += ns * qs * as * as / (BB * as * as + 1);
			c += ns * qs * as * as * as / (BB * as * as + 1);
		}

		a *= 0.5 * dt / epsilon0;
		b *= 0.5 * dt / epsilon0;
		c *= 0.5 * dt / epsilon0;
		a += 1;
	}

	VectorForm<0> Q(mesh);
	VectorForm<0> K(mesh);

	Q.Clear();
	K.Clear();

	Ev = MapTo<VERTEX>(E + dE * 0.5);

	for (auto &v : sp_list_)
	{
		auto & ns = v.second->n;
		auto & Js = v.second->J;
		Real ms = v.second->m;
		Real qs = v.second->q;
		Real as = (dt * qs) / (2.0 * ms);

		Q -= Js;
		K = Cross(Js, B0) * as + Ev * ns * qs * as + Js;
		Js = (K + Cross(K, B0) * as + B0 * (Dot(K, B0) * as * as)) / (BB * as * as + 1);
		Q -= Js;
	}

	Q *= 0.5 * dt / epsilon0;
	Q += Ev;
	Ev = (Q * a - Cross(Q, B0) * b + B0 * (Dot(Q, B0) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);

	for (auto &v : sp_list_)
	{
		auto & ns = v.second->n;
		auto & Js = v.second->J;
		auto ms = v.second->m;
		auto qs = v.second->q;

		Real as = (dt * qs) / (2.0 * ms);
		Js += (Ev + Cross(Ev, B0) * as + B0 * (Dot(Ev, B0) * as * as)) * (as * qs * ns) / (BB * as * as + 1);
	}

	dE = (MapTo<EDGE>(Ev) - E) + dE * 0.5;

	LOGGER << DONE;

}

template<typename TM>
template<typename TDict, typename ...Args>
void ColdFluidEM<TM>::Load(TDict const&dict, RForm<0> const & ne, Args const & ...)
{
	if (!dict)
		return;

	LOGGER << "Create ColdFluidEM solver";

	enableNonlinear_ = dict["Nonlinear"].template as<bool>(false);

	auto sp = dict["Species"];

	for (auto const & p : sp)
	{
		std::string key;

		if (!p.first.is_number())
		{
			key = p.first.template as<std::string>();
		}
		else
		{
			p.second.GetValue("Name", &key);
		}

		std::shared_ptr<Species> sp(
		        new Species(p.second["Mass"].template as<Real>(1.0), p.second["Charge"].template as<Real>(1.0), mesh));

		sp->n.Clear();

		if (ne.empty())
		{
			LoadField(p.second["Density"], &(sp->n));
		}
		else
		{
			sp->n = ne;
			if (p.second["Density"].is_number())
				sp->n *= p.second["Density"].template as<Real>(1.0);
		}

		sp->J.Clear();

		LoadField(p.second["Current"], &(sp->J));

		sp_list_.emplace(key, sp);

	}

	LOGGER << DONE;

}

template<typename TM>
void ColdFluidEM<TM>::DumpData(std::string const & path) const
{
	GLOBAL_DATA_STREAM.OpenGroup(path);

	for (auto const & p : sp_list_)
	{
		LOGGER << "Dump " << "n_" + p.first << " to "
		<< Dump(p.second->n.data(), "n_" + p.first, p.second->n.GetShape(), true);

		LOGGER << "Dump " << "J_" + p.first << " to "
		<< Dump(p.second->J.data(), "J_" + p.first, p.second->J.GetShape(), true);
	}
}

template<typename TM>
template<typename OS>
void ColdFluidEM<TM>::Print(OS & os) const
{
	os << "ColdFluid = { Nonlinear = " << std::boolalpha << enableNonlinear_ << ",\n"

	<< "  Species = { \n ";

	for (auto const & p : sp_list_)
	{
		os << "\n\t" << p.first

		<< " = { " << " Mass =" << p.second->m << "," << " Charge =" << p.second->q << ",\n"

		<< "\t n0 = " << Dump(p.second->n.data(), "n_" + p.first, p.second->n.GetShape(), false) << "\n"

		<< "\t J0 = " << Dump(p.second->J.data(), "J_" + p.first, p.second->J.GetShape(), false) << "\n"

		<< "\t},\n";
	}
	os << "\t}\n}";

}

template<typename OS, typename TM>
OS &operator<<(OS & os, ColdFluidEM<TM> const& self)
{
	self.Save(os);
	return os;
}
}  // namespace simpla

//
//template<typename TM>
//template<typename TE, typename TB> inline
//void ColdFluidEM<TM>::_NextTimeStepE2(Real dt, TE const &E, TB const &B, TE *dE)
//{
//	DEFINE_PHYSICAL_CONST(mesh.constants());
//
//	LOGGER << "Push Cold Fluid. Method2";
//
//	if (BB.empty())
//	{
//		MapTo(B, &B0);
//		BB = Dot(B0, B0);
//	}
//
//	if (Ev.empty())
//		MapTo(E, &Ev);
//
//	RForm<0> a(mesh);
//	RForm<0> b(mesh);
//	RForm<0> c(mesh);
//
//	a.Fill(0);
//	b.Fill(0);
//	c.Fill(0);
//
//	VectorForm<0> dEv(mesh);
//	MapTo(*dE, &dEv);
//
//	VectorForm<0> K(mesh);
//	K.Fill(0);
//
//	VectorForm<0> Q(mesh);
//	Q.Fill(0);
//
//	Ev += dEv * (0.5 * dt);
//
//	for (auto &v : sp_list_)
//	{
//		auto ms = v.second->m ;
//		auto qs = v.second->Z ;
//		auto & ns = v.second->n;
//		auto & Js = v.second->J;
//
//		Real as = 2.0 * ms / (dt * qs);
//
//		a += ns * qs / as;
//		b += ns * qs / (BB + as * as);
//		c += ns * qs / ((BB + as * as) * as);
//
//		Q -= Js;
//
//		K = Js * as + Cross(Js, B0) + Ev * (ns * qs);
//
////		Js = K / as + Cross(K, B0) / (BB + as * as) + Cross(Cross(K, B0), B0) / (as * (BB + as * as));
//		Js = (K * as * as + Cross(K, B0) * as + Dot(K, B0) * B0) / (as * (BB + as * as));
//
//		Q -= Js;
//	}
//
//	a *= (0.5 * dt) / epsilon0;
//	b *= (0.5 * dt) / epsilon0;
//	c *= (0.5 * dt) / epsilon0;
//
//	a += 1.0;
//
//	Q *= 0.5 * dt / epsilon0;
//
//	Q += Ev;
//
//	Ev = Q / a
//
//	- Cross(Q, B0) * b / ((c * BB - a) * (c * BB - a) + b * b * BB)
//
//	- Cross(Cross(Q, B0), B0) * (-c * c * BB + c * a - b * b) / (a * ((c * BB - a) * (c * BB - a) + b * b * BB))
//
//	;
//
//	for (auto &v : sp_list_)
//	{
//		auto ms = v.second->m ;
//		auto qs = v.second->Z ;
//		auto & ns = v.second->n;
//		auto & Js = v.second->J;
//		Real as = 2.0 * ms / (dt * qs);
//
////		Js += (Ev / as + Cross(Ev, B0) / (BB + as * as) + Cross(Cross(Ev, B0), B0) / (as * (BB + as * as))) * (ns * qs);
//
//		Js += (Ev * as * as + Cross(Ev, B0) * as + Dot(K, B0) * B0) * (ns * qs / (as * (BB + as * as)));
//	}
//
//	Ev += dEv * (0.5 * dt);
//
//	MapTo(Ev, dE);
//
//	*dE -= E;
//	*dE /= dt;
//}

/**
 *
 *
 *
 //template<typename TL> inline auto operator*(nTuple<3, TL> const & l, Real r)
 //-> nTuple<3, decltype(l[0]*r)>
 //{
 //	nTuple<3, decltype(l[0]*r)> res = { l[0] * r, l[1] * r, l[2] * r };
 //
 //	return std::move(res);
 //}
 namespace fetl_impl
 {

 template<typename TM, typename TL, typename TR, typename ...TI>
 inline auto FieldOpEval(Int2Type<MULTIPLIES>, Field<Geometry<TM, 0>, TL> const &l, Field<Geometry<TM, 0>, TR> r,
 TI ... s)
 DECL_RET_TYPE((l.get(s...)*r.get(s...)))

 template<typename TM, typename TL, typename ...TI>
 inline auto FieldOpEval(Int2Type<MULTIPLIES>, Field<Geometry<TM, 0>, TL> const &l, Real r, TI ... s)
 DECL_RET_TYPE((l.get(s...)*r))

 }

 //template<typename TL, typename TR> inline auto Cross(nTuple<3, TL> const & l, nTuple<3, TR> const & r)
 //->nTuple<3,decltype(l[0]*r[0])>
 //{
 //	nTuple<3, decltype(l[0]*r[0])> res = {
 //
 //	l[1] * r[2] - l[2] * r[1],
 //
 //	l[2] * r[0] - l[0] * r[2],
 //
 //	l[0] * r[1] - l[1] * r[0]
 //
 //	};
 //	return std::move(res);
 //}
 */

#endif /* COLD_FLUID_H_ */
