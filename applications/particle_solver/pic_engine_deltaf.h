/*
 * pic_engine_deltaf.h
 *
 *  Created on: 2013年12月10日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_DELTAF_H_
#define PIC_ENGINE_DELTAF_H_

#include <cstddef>
#include <sstream>
#include <string>

#include "../../src/utilities/ntuple.h"
#include "../../src/utilities/primitives.h"
#include "../../src/physics/physical_constants.h"
#include "../../src/io/hdf5_datatype.h"

namespace simpla
{

template<typename TM, typename Interpolator = typename TM::interpolator_type>
struct PICEngineDeltaF
{

public:
	enum
	{
		EnableImplicit = false
	};
	Real m;
	Real q;

	typedef PICEngineDeltaF<TM, Interpolator> this_type;
	typedef TM mesh_type;
	typedef Interpolator interpolator_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type:: template field<VERTEX, scalar_type> n_type;

	typedef typename std::conditional<EnableImplicit,
	        typename mesh_type::template field<VERTEX, nTuple<3, scalar_type>>,
	        typename mesh_type::template field<EDGE, scalar_type> >::type J_type;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
		scalar_type w;
	};

	typedef std::tuple<coordinates_type, Vec3, Real, scalar_type> compact_point_type;

	auto Compact(Point_s && p) DECL_RET_TYPE(std::tie(p.x,p.v,p.f,p.w))

	auto Decompact(compact_point_type && p) DECL_RET_TYPE(Point_s(
					{	std::get<0>(p),std::get<1>(p),std::get<2>(p),std::get<3>(p.w)}))
private:
	Real cmr_, q_kT_;
public:
	mesh_type const &mesh;

	PICEngineDeltaF(mesh_type const &m)
			: mesh(m), m(1.0), q(1.0), cmr_(1.0), q_kT_(1.0)
	{
	}
	template<typename ...Others>
	PICEngineDeltaF(mesh_type const &pmesh, Others && ...others)
			: PICEngineDeltaF(pmesh)
	{
		Load(std::forward<Others >(others)...);
	}
	template<typename TDict, typename ...Args>
	void Load(TDict const& dict, Args const & ...args)
	{
		m = (dict["Mass"].template as<Real>(1.0));
		q = (dict["Charge"].template as<Real>(1.0));

		cmr_ = (q / m);

		DEFINE_PHYSICAL_CONST

		q_kT_ = q / (dict["Temperature"].template as<Real>(1.0) * boltzmann_constant);

		{
			std::ostringstream os;
			os

			<< "H5T_COMPOUND {          "

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"x\" : " << (offsetof(Point_s, x)) << ";"

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"v\" :  " << (offsetof(Point_s, v)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"f\" : " << (offsetof(Point_s, f)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"w\" : " << (offsetof(Point_s, w)) << ";"

			<< "}";

			GLOBAL_HDF5_DATA_TYPE_FACTORY.template Register < Point_s > (os.str());
		}

	}
	~PICEngineDeltaF()
	{
	}

	static std::string GetTypeAsString()
	{
		return "DeltaF";
	}

	std::string Save(std::string const & path = "", bool is_verbose = false) const
	{
		std::stringstream os;

		DEFINE_PHYSICAL_CONST
		;

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m / proton_mass << " * m_p"

		<< " , " << "Charge = " << q / elementary_charge << " * q_e"

		<< " , " << "Temperature = " << q / q_kT_ / elementary_charge << "* eV"

		;

		return os.str();
	}

	static Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		p.w = 0.0;
		return std::move(p);
	}
	template<typename TJ, typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Point_s * p, Real dt, TJ *J, TE const &fE, TB const & fB,
			Others const &...others) const
	{
		p->x += p->v * dt * 0.5;
		auto cE = interpolator_type::GatherCartesian( fE, p->x);
		auto B = real(interpolator_type::GatherCartesian(fB, p->x));
		auto E = real(cE);
		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);

		p->v += v_;
		auto a = (-Dot(cE, p->v) * q_kT_ * dt);
		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

		p->v += v_;
		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt * 0.5;

		interpolator_type::ScatterCartesian( J,std::make_tuple(p->x,p->v), p->f * p->w * q);

	}
	template<typename TE, typename TB, typename ... Others>
	inline void NextTimeStepHalf(Point_s * p, Real dt, TE const &fE, TB const & fB, Others const &...others) const
	{
	}
//	// x(-1/2->1/2), w(-1/2,1/2)
//	template<typename TJ, typename TE, typename TB, typename ... Others>
//	inline void NextTimeStepZero(Bool2Type<true>, Point_s * p, Real dt, TJ *J,
//			TE const &fE, TB const & fB, Others const &...others) const
//	{
//		p->x += p->v * dt * 0.5;
//
////		auto B = interpolator_type::Gather(fB, p->x);
//		auto E = interpolator_type::Gather(fE, p->x);
//
//		auto a = (-Dot(E, p->v) * q_kT_ * dt);
//		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);
//
//		p->x += p->v * dt * 0.5;
//
//		Vec3 v;
//		v = p->v * p->f * p->w * q;
//		interpolator_type::Scatter(p->x, v, J);
//
//	}
//	template<typename TE, typename TB, typename ... Others>
//	inline void NextTimeStepHalf(Bool2Type<true>, Point_s * p, Real dt,
//			TE const &fE, TB const & fB, Others const &...others) const
//	{
//
//		auto B = interpolator_type::Gather(fB, p->x);
//		auto E = interpolator_type::Gather(fE, p->x);
//
//		Vec3 v_;
//
//		auto t = B * (cmr_ * dt * 0.5);
//
//		p->v += E * (cmr_ * dt * 0.5);
//
//		v_ = p->v + Cross(p->v, t);
//
//		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);
//
//		p->v += v_ * 2.0;
//
//		p->v += E * (cmr_ * dt * 0.5);
//
//	}
	template<typename TV, typename ...Args>
	void Scatter(Point_s const & p, typename mesh_type:: template field < VERTEX, TV> * n, Args const & ...) const
	{
		interpolator_type::ScatterCartesian( n,std::make_tuple(p.x, q * p.f * p.w));
	}
	inline Real PullBack(Point_s const & p, nTuple<3, Real> *x, nTuple<3, Real> * v) const
	{
		*x = p.x;
		*v = p.v;
		return p.f * p.w;
	}
	inline void PushForward(nTuple<3, Real> const&x, nTuple<3, Real> const& v, Point_s * p) const
	{
		p->x = x;
		p->v = v;
	}
	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s(
						{	x, v, f, 0}));
	}

};

template<typename ... TS> std::ostream&
operator<<(std::ostream& os, typename PICEngineDeltaF<TS...>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " , w=" << p.w << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_DELTAF_H_ */
