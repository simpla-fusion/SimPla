/*
 * rect_mesh.h
 *
 *  Created on: 2014年2月26日
 *      Author: salmon
 */

#ifndef RECT_MESH_H_
#define RECT_MESH_H_

#include "../fetl/primitives.h"
#include "../utilities/type_utilites.h"
#include "../utilities/utilities.h"
#include "../utilities/memory_pool.h"
#include "../physics/physical_constants.h"
#include "octree_forest.h"
namespace simpla
{
struct EuclideanSpace
{
	static constexpr int NDIMS = 3;
	typedef nTuple<NDIMS, Real> vector_type;
	typedef nTuple<NDIMS, Real> covector_type;
	typedef nTuple<NDIMS, Real> coordinates_type;
	typedef unsigned long index_type;

	constexpr Real g_t[NDIMS][NDIMS] = {

	1, 0, 0,

	0, 1, 0,

	0, 0, 1

	};

	template<int I, int J>
	constexpr Real g(index_type const &) const
	{
		return g_t[I][J];
	}

	//! determind of metric tensor gd=|g|
	constexpr Real gd(index_type const &) const
	{
		return 1.0;
	}
	Real g_(index_type const & s, unsigned int a, unsigned int b) const
	{
		return g_t[a][b];
	}

	//! diagonal term of metric tensor
	Real g_(index_type const & s, unsigned int l) const
	{
		return g_t[l][l];
	}

	coordinates_type const & Trans(coordinates_type const & x) const
	{
		return x;
	}
	coordinates_type const & InvTrans(coordinates_type const & x) const
	{
		return x;
	}
	template<typename index_type>
	vector_type PullBack(index_type const & s, vector_type const & v) const
	{
		return v;
	}

	template<typename index_type>
	vector_type PushForward(index_type const & s, vector_type const & v) const
	{
		return v;
	}

};
template<typename Metric = EuclideanSpace>
class RectMesh: public OcForest, public Metric
{
public:
	typedef OcForest base_type;
	typedef RectMesh this_type;

	static constexpr unsigned int NDIMS = 3;

	RectMesh();
	~RectMesh();

	template<typename TDict>
	RectMesh(TDict const & dict)
			: OcForest(dict), Metric(dict)
	{
		Load(dict);
	}
	template<typename TDict>
	void Load(TDict const & dict)
	{
	}

	this_type & operator=(const this_type&) = delete;

	//***************************************************************************************************
	// Geometric properties
	// Metric
	//***************************************************************************************************

	typedef nTuple<3, Real> coordinates_type;

	coordinates_type xmin_ = { 0, 0, 0 };

	coordinates_type xmax_ = { 1, 1, 1 };

	coordinates_type inv_L = { 1.0, 1.0, 1.0 };

	template<int IN, typename T>
	inline void SetExtent(nTuple<IN, T> const & pmin, nTuple<IN, T> const & pmax)
	{
		int n = IN < NUM_OF_DIMS ? IN : NUM_OF_DIMS;

		for (int i = 0; i < n; ++i)
		{
			xmin_[i] = pmin[i];
			xmax_[i] = pmax[i];
		}

		for (int i = n; i < NUM_OF_DIMS; ++i)
		{
			xmin_[i] = 0;
			xmax_[i] = 0;
		}
	}

	inline std::pair<coordinates_type, coordinates_type> GetExtent() const
	{
		return std::move(std::make_pair(xmin_, xmax_));
	}

	inline coordinates_type GetCoordinates(index_type const &s) const
	{
		coordinates_type res;
		res = xmin_ + (xmax_ - xmin_) * base_type::GetCoordinates(s);
		return std::move(res);
	}
	Real idh(index_type s, int N) const
	{
		return static_cast<Real>(1UL << (INDEX_DIGITS - index_digits_[N]));
	}

	//***************************************************************************************************
	// Exterior algebra
	//***************************************************************************************************

	//! Form<IR> ^ Form<IR> => Form<IR+IL>
	template<int IL, int IR, typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,
	        Field<Geometry<this_type, IL>, TL> const &l, Field<Geometry<this_type, IR>, TR> const &r,
	        index_type s) const ->decltype(l[s]*r[s])
	{
		return Wedge_(l, r, s);
	}

	template<int IL, typename TL> inline auto OpEval(Int2Type<HODGESTAR>, Field<Geometry<this_type, IL>, TL> const & f,
	        index_type s) const->decltype(f[s])
	{
		return HodgeStar_(f, s);
	}
	template<int IL, typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,
	        Field<Geometry<this_type, IL>, TL> const & f, index_type s)
	{
		return ExteriorDerivative_(f, s);
	}
	template<int IL, typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,
	        Field<Geometry<this_type, IL>, TL> const & f, index_type s)
	{
		return Codifferential_(f, s);
	}

	template<int IL, typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>,
	        nTuple<NDIMS, TR> const & v, Field<Geometry<this_type, IL>, TL> const & f, index_type s)
	{
		return InteriorProduct_(v, f, s);
	}

private:
//! Form<IR> ^ Form<IR> => Form<IR+IL>
	template<int IL, int IR, typename TL, typename TR> inline auto Wedge_(Field<Geometry<this_type, IL>, TL> const &l,
	        Field<Geometry<this_type, IR>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		return Dot(mapto(l, Int2Type<IL + IR>(), s), mapto(r, Int2Type<IL + IR>(), s));
	}

	template<int IL, typename TL> inline auto HodgeStar_(Field<Geometry<this_type, IL>, TL> const & f,
	        index_type s) const->decltype(f[s])
	{
		return mapto(f, Int2Type<NDIMS - IL>(), s);
	}

	template<typename TL> inline auto ExteriorDerivative_(Field<Geometry<this_type, VERTEX>, TL> const & f,
	        index_type s)->decltype(f[s])
	{
		auto d = s & (_MA >> (s.H + 1));

		unsigned int n = _N(s);

		return (f[s + d] - f[s - d]);
	}

	template<typename TL> inline auto ExteriorDerivative_(Field<Geometry<this_type, EDGE>, TL> const & f,
	        index_type s)->decltype(f[s])
	{
		auto Y = _R(_I(s)) & (_MA >> (s.H + 1));
		auto Z = _RR(_I(s)) & (_MA >> (s.H + 1));

		return (f[s + Y] - f[s - Y]) - (f[s + Z] - f[s - Z]);
	}

	template<typename TL> inline auto ExteriorDerivative_(Field<Geometry<this_type, FACE>, TL> const & f,
	        index_type s)->decltype(f[s])
	{
		auto X = (_MI >> (s.H + 1));
		auto Y = (_MJ >> (s.H + 1));
		auto Z = (_MK >> (s.H + 1));

		return (f[s + X] - f[s - X]) + (f[s + Y] - f[s - Y]) + (f[s + Z] - f[s - Z]);
	}

	template<int IL, typename TL> inline auto ExteriorDerivative_(Field<Geometry<this_type, IL>, TL> const & f,
	        index_type s)->typename std::enable_if<IL>=NDIMS, decltype(f[s])>::type
	{
		return 0;
	}

	template<int IL, typename TL> inline auto Codifferential_(Field<Geometry<this_type, IL>, TL> const & f,
	        index_type s)->typename std::enable_if<IL==0, decltype(f[s])>::type
	{
		return 0;
	}

	template<int IL, typename TL> inline auto Codifferential_(Field<Geometry<this_type, EDGE>, TL> const & f,
	        index_type s)->decltype(f[s])
	{
		auto X = (_MI >> (s.H + 1));
		auto Y = (_MJ >> (s.H + 1));
		auto Z = (_MK >> (s.H + 1));

		return (f[s + X] - f[s - X]) + (f[s + Y] - f[s - Y]) + (f[s + Z] - f[s - Z]);
	}

	template<typename TL> inline auto Codifferential_(Field<Geometry<this_type, FACE>, TL> const & f,
	        index_type s)->decltype(f[s])
	{
		auto Y = _R(_I(s)) & (_MA >> (s.H + 1));
		auto Z = _RR(_I(s)) & (_MA >> (s.H + 1));

		return (f[s + Y] - f[s - Y]) - (f[s + Z] - f[s - Z]);
	}
	template<typename TL> inline auto Codifferential_(Field<Geometry<this_type, VOLUME>, TL> const & f,
	        index_type s)->decltype(f[s])
	{
		auto d = _I(s) & (_MA >> (s.H + 1));

		unsigned int n = _N(_I(s));

		return (f[s + d] - f[s - d]);
	}

	template<typename TL, typename TR> inline auto InteriorProduct_(nTuple<NDIMS, TR> const & v,
	        Field<Geometry<this_type, VERTEX>, TL> const & f, index_type s)->decltype(f[s]*v[0])
	{
		return 0;
	}

	template<typename TL, typename TR> inline auto InteriorProduct_(nTuple<NDIMS, TR> const & v,
	        Field<Geometry<this_type, EDGE>, TL> const & f, index_type s)->decltype(f[s]*v[0])
	{
		return Dot(mapto(f, Int2Typ2<VERTEX>, s), v);
	}

	template<typename TL, typename TR> inline auto InteriorProduct_(nTuple<NDIMS, TR> const & v,
	        Field<Geometry<this_type, FACE>, TL> const & f, index_type s)->decltype(f[s]*v[0])
	{
		unsigned int n = _N(s);
		auto Y = _R(s) & (_MA >> (s.H + 1));
		auto Z = _RR(s) & (_MA >> (s.H + 1));
		return

		(f[s + Y] + f[s - Y]) * 0.5 * v[(n + 2) % 3] -

		(f[s + Z] + f[s - Z]) * 0.5 * v[(n + 1) % 3];
	}
	template<typename TL, typename TR> inline auto InteriorProduct_(nTuple<NDIMS, TR> const & v,
	        Field<Geometry<this_type, VOLUME>, TL> const & f, index_type s)->decltype(f[s]*v[0])
	{
		unsigned int n = _N(_I(s));
		return mapto(f, Int2Type<FACE>(), s) * v[n];
	}

	//***************************************************************************************************
	// Map
	//***************************************************************************************************
	/**
	 *  Mapto -
	 *    mapto(Int2Type<VERTEX> ,   //tarGet topology position
	 *     Field<this_type,1 , TExpr> const & vl,  //field
	 *      SizeType s   //grid index of point
	 *      )
	 * target topology position:
	 *             z 	y 	x
	 *       000 : 0,	0,	0   vertex
	 *       001 : 0,	0,1/2   edge
	 *       010 : 0, 1/2,	0
	 *       100 : 1/2,	0,	0
	 *       011 : 0, 1/2,1/2   Face
	 * */

	template<typename T, int IF> inline auto	//
	mapto(T const &f, Int2Type<IF> , ...) const
	ENABLE_IF_DECL_RET_TYPE(is_primitive<T>::value,f)

	//!  n => n
	template<int IL, typename TL> inline typename Field<Geometry<this_type, IF>, TL>::value_type mapto(
	        Field<Geometry<this_type, IF>, IL> const &f, Int2Type<IL>, index_type s) const
	{
		return l[s];
	}

	//!  1 => 0
	template<typename TL> inline nTuple<3, typename Field<Geometry<this_type, EDGE>, TL>::value_type> mapto(
	        Field<Geometry<this_type, EDGE>, TL> const &f, Int2Type<VERTEX>, index_type s) const
	{
		auto X = _MI >> (s.H + 1);
		auto Y = _MJ >> (s.H + 1);
		auto Z = _MK >> (s.H + 1);

		return nTuple<3, typename Field<Geometry<this_type, EDGE>, TL>::value_type>(

		{

		(f[s - X] + f[s + X]) * 0.5,

		(f[s - Y] + f[s + Y]) * 0.5,

		(f[s - Z] + f[s + Z]) * 0.5

		}

		);
	}
	//!  2 => 3
	template<typename TL> inline nTuple<3, typename Field<Geometry<this_type, FACE>, TL>::value_type> mapto(
	        Field<Geometry<this_type, FACE>, TL> const &f, Int2Type<VOLUME>, index_type s) const
	{
		auto X = _MI >> (s.H + 1);
		auto Y = _MJ >> (s.H + 1);
		auto Z = _MK >> (s.H + 1);

		return nTuple<3, typename Field<Geometry<this_type, EDGE>, TL>::value_type>(

		{

		(f[s - X] + f[s + X]) * 0.5,

		(f[s - Y] + f[s + Y]) * 0.5,

		(f[s - Z] + f[s + Z]) * 0.5

		}

		);
	}

	//!  2=> 0
	template<typename TL> inline nTuple<3, typename Field<Geometry<this_type, FACE>, TL>::value_type> mapto(
	        Field<Geometry<this_type, FACE>, TL> const &f, Int2Type<VERTEX>, index_type s) const
	{
		auto X = _MI >> (s.H + 1);
		auto Y = _MJ >> (s.H + 1);
		auto Z = _MK >> (s.H + 1);

		return nTuple<3, typename Field<Geometry<this_type, IF>, TL>::value_type>(

		{

		(f[(s - Y) - Z] + f[(s - Y) + Z] + f[(s + Y) - Z] + f[(s + Y) + Z]) * 0.25,

		(f[(s - Z) - X] + f[(s - Z) + X] + f[(s + Z) - X] + f[(s + Z) + X]) * 0.25,

		(f[(s - X) - Y] + f[(s - X) + Y] + f[(s + X) - Y] + f[(s + X) + Y]) * 0.25

		}

		);
	}

	//!  1 => 3
	template<typename TL> inline nTuple<3, typename Field<Geometry<this_type, EDGE>, TL>::value_type> mapto(
	        Field<Geometry<this_type, EDGE>, TL> const &f, Int2Type<VOLUME>, index_type s) const
	{
		auto X = _MI >> (s.H + 1);
		auto Y = _MJ >> (s.H + 1);
		auto Z = _MK >> (s.H + 1);

		return nTuple<3, typename Field<Geometry<this_type, IF>, TL>::value_type>(

		{

		(f[(s - Y) - Z] + f[(s - Y) + Z] + f[(s + Y) - Z] + f[(s + Y) + Z]) * 0.25,

		(f[(s - Z) - X] + f[(s - Z) + X] + f[(s + Z) - X] + f[(s + Z) + X]) * 0.25,

		(f[(s - X) - Y] + f[(s - X) + Y] + f[(s + X) - Y] + f[(s + X) + Y]) * 0.25

		}

		);
	}

	//!  3 => 0
	template<typename TL> inline nTuple<3, typename Field<Geometry<this_type, VOLUME>, TL>::value_type> mapto(
	        Field<Geometry<this_type, VOLUME>, TL> const &f, Int2Type<VERTEX>, index_type s) const
	{
		auto X = _MI >> (s.H + 1);
		auto Y = _MJ >> (s.H + 1);
		auto Z = _MK >> (s.H + 1);
		return (

		f[((s - X) - Y) - Z] + f[((s - X) - Y) + Z] + f[((s - X) + Y) - Z] + f[((s - X) + Y) + Z] +

		f[((s + X) - Y) - Z] + f[((s + X) - Y) + Z] + f[((s + X) + Y) - Z] + f[((s + X) + Y) + Z]

		) * 0.125;
	}
	//!  0 => 3
	template<typename TL> inline typename Field<Geometry<this_type, VERTEX>, TL>::value_type mapto(
	        Field<Geometry<this_type, VERTEX>, TL> const &f, Int2Type<VOLUME>, index_type s) const
	{
		auto X = _MI >> (s.H + 1);
		auto Y = _MJ >> (s.H + 1);
		auto Z = _MK >> (s.H + 1);
		return (

		f[((s - X) - Y) - Z] + f[((s - X) - Y) + Z] + f[((s - X) + Y) - Z] + f[((s - X) + Y) + Z] +

		f[((s + X) - Y) - Z] + f[((s + X) - Y) + Z] + f[((s + X) + Y) - Z] + f[((s + X) + Y) + Z]

		) * 0.125;
	}

	//!  0 => 1
	template<typename TL> inline typename Field<Geometry<this_type, VERTEX>, TL>::value_type mapto(
	        Field<Geometry<this_type, VERTEX>, TL> const &l, Int2Type<EDGE>, index_type s) const
	{
		auto D = s & (_MA >> (s.H + 1));

		return (f[s - D] + f[s + D]) * 0.5;
	}

	//!  3 => 2
	template<typename TL> inline typename Field<Geometry<this_type, VOLUME>, TL>::value_type mapto(
	        Field<Geometry<this_type, VOLUME>, TL> const &f, Int2Type<FACE>, index_type s) const
	{
		auto D = _I(s) & (_MA >> (s.H + 1));

		return (f[s - D] + f[s + D]) * 0.5;
	}

	//!  2 => 1
	template<typename TL> inline typename Field<Geometry<this_type, FACE>, TL>::value_type mapto(
	        Field<Geometry<this_type, VOLUME>, TL> const &f, Int2Type<EDGE>, index_type s) const
	{
		s = _I(s);
		auto X = _MI >> (s.H + 1);
		auto Y = _MJ >> (s.H + 1);
		auto Z = _MK >> (s.H + 1);

		return (

		f[((s - X) - Y) - Z] + f[((s - X) - Y) + Z] + f[((s - X) + Y) - Z] + f[((s - X) + Y) + Z] +

		f[((s + X) - Y) - Z] + f[((s + X) - Y) + Z] + f[((s + X) + Y) - Z] + f[((s + X) + Y) + Z]

		) * 0.125;
	}

	//!  1 => 2
	template<typename TL> inline typename Field<Geometry<this_type, EDGE>, TL>::value_type mapto(
	        Field<Geometry<this_type, EDGE>, TL> const &f, Int2Type<FACE>, index_type s) const
	{
		s = _I(s);
		auto X = _MI >> (s.H + 1);
		auto Y = _MJ >> (s.H + 1);
		auto Z = _MK >> (s.H + 1);

		return (

		f[((s - X) - Y) - Z] + f[((s - X) - Y) + Z] + f[((s - X) + Y) - Z] + f[((s - X) + Y) + Z] +

		f[((s + X) - Y) - Z] + f[((s + X) - Y) + Z] + f[((s + X) + Y) - Z] + f[((s + X) + Y) + Z]

		) * 0.125;
	}

	//!  3 => 1
	template<typename TL> inline typename Field<Geometry<this_type, VOLUME>, TL>::value_type mapto(
	        Field<Geometry<this_type, VOLUME>, TL> const &f, Int2Type<EDGE>, index_type s) const
	{
		auto Y = _R(s) & (_MA >> (s.H + 1));
		auto Z = _RR(s) & (_MA >> (s.H + 1));

		return (f[(s - Y) - Z] + f[(s - Y) + Z] + f[(s + Y) - Z] + f[(s + Y) + Z]) * 0.25;

	}

	//!  0 => 2
	template<typename TL> inline typename Field<Geometry<this_type, VERTEX>, TL>::value_type mapto(
	        Field<Geometry<this_type, VERTEX>, TL> const &f, Int2Type<FACE>, index_type s) const
	{
		auto Y = _R(_I(s)) & (_MA >> (s.H + 1));
		auto Z = _RR(_I(s)) & (_MA >> (s.H + 1));

		return (f[(s - Y) - Z] + f[(s - Y) + Z] + f[(s + Y) - Z] + f[(s + Y) + Z]) * 0.25;
	}

};
//***************************************************************************************************
//*	Miscellaneous
//***************************************************************************************************

//***************************************************************************************************
//* Media Tags
//***************************************************************************************************
//
//private:
//
//	MediaTag<this_type> tags_;
//public:
//
//	typedef typename MediaTag<this_type>::tag_type tag_type;
//	MediaTag<this_type> & tags()
//	{
//		return tags_;
//	}
//	MediaTag<this_type> const& tags() const
//	{
//
//		return tags_;
//	}
//
//	//***************************************************************************************************
//	//* Configure
//	//***************************************************************************************************
//
//	template<typename TDict> void Load(TDict const &cfg);
//
//	std::ostream & Save(std::ostream &vm) const;
//
//	void Update();
//
//	inline bool operator==(this_type const & r) const
//	{
//		return (this == &r);
//	}
//
//
//
//	//***************************************************************************************************
//	//* Time
//	//***************************************************************************************************
//
//private:
//	Real dt_ = 0.0; //!< time step
//	Real time_ = 0.0;
//public:
//
//	void NextTimeStep()
//	{
//		time_ += dt_;
//	}
//	Real GetTime() const
//	{
//		return time_;
//	}
//
//	void GetTime(Real t)
//	{
//		time_ = t;
//	}
//	inline Real GetDt() const
//	{
//		CheckCourant();
//		return dt_;
//	}
//
//	inline void SetDt(Real dt = 0.0)
//	{
//		dt_ = dt;
//		Update();
//	}
//	double CheckCourant() const
//	{
//		DEFINE_GLOBAL_PHYSICAL_CONST
//
//		nTuple<3, Real> inv_dx_;
//		inv_dx_ = 1.0 / GetDx() / (xmax_ - xmin_);
//
//		Real res = 0.0;
//
//		for (int i = 0; i < 3; ++i)
//			res += inv_dx_[i] * inv_dx_[i];
//
//		return std::sqrt(res) * speed_of_light * dt_;
//	}
//
//	void FixCourant(Real a)
//	{
//		dt_ *= a / CheckCourant();
//	}
//
//	//***************************************************************************************************
//	//* Container: storage depend
//	//***************************************************************************************************
//
//	template<typename TV> using Container=std::shared_ptr<TV>;
//
//	nTuple<NUM_OF_DIMS, size_type> strides_ = { 0, 0, 0 };
//
//	inline nTuple<NUM_OF_DIMS, index_type> const & GetStrides() const
//	{
//		return strides_;
//	}
//
//	inline index_type GetNumOfElements(int iform) const
//	{
//
//		return (num_grid_points_ * num_comps_per_cell_[iform]);
//	}
//
//	template<int iform, typename TV> inline Container<TV> MakeContainer() const
//	{
//		return (MEMPOOL.allocate_shared_ptr < TV > (GetNumOfElements(iform)));
//	}
}
// namespace simpla

#endif /* RECT_MESH_H_ */
