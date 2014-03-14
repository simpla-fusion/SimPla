/*
 * pic_engine_ggauge.h
 *
 *  Created on: 2013年10月23日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_GGAUGE_H_
#define PIC_ENGINE_GGAUGE_H_

#include "../fetl/fetl.h"
#include "../fetl/ntuple.h"
#include "../physics/constants.h"
namespace simpla
{
template<typename TM, int NMATE = 8, typename TScaler = Real>
class PICEngineGGauge
{

public:
	typedef PICEngineGGauge<TM, NMATE, TScaler> this_type;
	typedef TM mesh_type;
	typedef TScaler scalar_type;
private:
	Real m_, q_, cmr_, T_, vT_;
	Real cosdq[NMATE], sindq[NMATE];
public:
	mesh_type const &mesh;

	DEFINE_FIELDS(mesh_type)

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
		nTuple<NMATE, scalar_type> w;

		static std::string DataTypeDesc()
		{
			std::ostringstream os;
			os

			<< "H5T_COMPOUND {          "

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"x\" : " << (offsetof(Point_s, x)) << ";"

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"v\" :  " << (offsetof(Point_s, v)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"f\" : " << (offsetof(Point_s, f)) << ";"

			<< "   H5T_ARRAY { [" << NMATE << "] H5T_NATIVE_DOUBLE}    \"w\" :  " << (offsetof(Point_s, w)) << ";"

			<< "}";

			return os.str();
		}
	};

	PICEngineGGauge(mesh_type const &pmesh) :
			mesh(pmesh), m_(1.0), q_(1.0), cmr_(1.0), T_(1.0), vT_(1.0)
	{

	}
	~PICEngineGGauge()
	{
	}

	size_t GetAffectedRegion() const
	{
		return 4;
	}
	static inline std::string TypeName()
	{
		return "GGauge" + ToString(NMATE);
	}
	std::string GetTypeAsString() const
	{
		return TypeName();
	}

	void Load(LuaObject const &obj)
	{

		obj["T"].as<Real>(&T_);

		Update();
	}
	void Update()
	{
		cmr_ = q_ / m_;

		vT_ = std::sqrt(2.0 * T_ / m_);

		constexpr Real theta = TWOPI / static_cast<Real>(NMATE);

		for (int i = 0; i < NMATE; ++i)
		{
			sindq[i] = std::sin(theta * i);
			cosdq[i] = std::cos(theta * i);
		}

	}
	std::ostream & Save(std::ostream & os) const
	{

		os << "Engine = 'GGague" << NMATE << "' " << " , ";

		return os;
	}

	Real GetMass() const
	{
		return m_;
	}

	Real GetCharge() const
	{
		return q_;
	}
	static Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		return std::move(p);
	}

	template<typename TB, typename TE, typename ... Others>
	inline void NextTimeStep(Point_s * p, Real dt, TB const & B, TE const &E, Others const &...others) const
	{
		RVec3 B0 = real(B(p->x));
		Real BB = Dot(B0, B0);

		RVec3 b = B0 / std::sqrt(BB);
		Vec3 v0, v1, r0, r1;
		Vec3 Vc;
		Vc = (Dot(p->v, B0) * B0) / BB;
		v1 = Cross(p->v, b);
		v0 = -Cross(v1, b);
		r0 = -Cross(v0, B0) / (cmr_ * BB);
		r1 = -Cross(v1, B0) / (cmr_ * BB);

		for (int ms = 0; ms < NMATE; ++ms)
		{
			Vec3 v, r;
			v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
			r = (p->x + r0 * cosdq[ms] + r1 * sindq[ms]);
			p->w[ms] += 0.5 * Dot(E(r), v) * dt;
		}

		Vec3 t, V_;

		t = B0 * cmr_ * dt * 0.5;

		V_ = p->v + Cross(p->v, t);

		p->v += Cross(V_, t) / (Dot(t, t) + 1.0) * 2.0;

		Vc = (Dot(p->v, B0) * B0) / BB;

		p->x += Vc * dt * 0.5;

		v1 = Cross(p->v, b);
		v0 = -Cross(v1, b);
		r0 = -Cross(v0, B0) / (cmr_ * BB);
		r1 = -Cross(v1, B0) / (cmr_ * BB);

		for (int ms = 0; ms < NMATE; ++ms)
		{
			Vec3 v, r;
			v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
			r = (p->x + r0 * cosdq[ms] + r1 * sindq[ms]);
			p->w[ms] += 0.5 * Dot(E(r), v) * q_ / T_ * dt;

		}
		p->x += Vc * dt * 0.5;
	}

	template<typename TV, typename TB, typename ... Others>
	inline typename std::enable_if<!is_ntuple<TV>::value, void>::type Scatter(Point_s const &p,
			Field<mesh_type, VERTEX, TV>* n, TB const & B, Others const &... others) const
	{
		RVec3 B0 = real(B(p.x));
		Real BB = Dot(B0, B0);

		RVec3 b = B0 / std::sqrt(BB);

		Vec3 v0, v1, r0, r1;
		Vec3 Vc;

		Vc = (Dot(p.v, B0) * B0) / BB;

		v1 = Cross(p.v, b);
		v0 = -Cross(v1, b);
		r0 = -Cross(v0, B0) / (cmr_ * BB);
		r1 = -Cross(v1, B0) / (cmr_ * BB);

		for (int ms = 0; ms < NMATE; ++ms)
		{
			Vec3 v, r;
			r = (p.x + r0 * cosdq[ms] + r1 * sindq[ms]);

			mesh.Scatter(r, p.w[ms], n);
		}

	}

	template<int IFORM, typename TV, typename TB, typename ...Others>
	inline void Scatter(Point_s const &p, Field<mesh_type, IFORM, TV>* J, TB const & B, Others const &... others) const
	{
		RVec3 B0 = real(B(p.x));
		Real BB = Dot(B0, B0);

		RVec3 b = B0 / sqrt(BB);
		Vec3 v0, v1, r0, r1;
		Vec3 Vc;

		Vc = (Dot(p.v, B0) * B0) / BB;

		v1 = Cross(p.v, b);
		v0 = -Cross(v1, b);
		r0 = -Cross(v0, B0) / (cmr_ * BB);
		r1 = -Cross(v1, B0) / (cmr_ * BB);
		for (int ms = 0; ms < NMATE; ++ms)
		{
			Vec3 v, r;
			v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
			r = (p.x + r0 * cosdq[ms] + r1 * sindq[ms]);

			mesh.Scatter(r, v * p.w[ms] * p.f, J);
		}
	}

	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		Point_s res;
		res.x = x;
		res.v = v;
		res.f = f;
		res.w *= 0;
		return std::move(res);
	}

}
;

template<typename TM> std::ostream&
operator<<(std::ostream& os, typename PICEngineGGauge<TM>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " , w=" << p.w << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_GGAUGE_H_ */
