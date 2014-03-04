/*
 * rect_mesh.h
 *
 *  Created on: 2014年2月26日
 *      Author: salmon
 */

#ifndef RECT_MESH_H_
#define RECT_MESH_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>
#include <memory>

#include "../fetl/field.h"
#include "../fetl/ntuple_ops.h"
#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../modeling/media_tag.h"
#include "../physics/physical_constants.h"
#include "../utilities/memory_pool.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/type_utilites.h"
//#include "../utilities/utilities.h"
#include "octree_forest.h"

namespace simpla
{
struct EuclideanSpace
{
	static constexpr int NDIMS = 3;
	typedef nTuple<NDIMS, Real> vector_type;
	typedef nTuple<NDIMS, Real> covector_type;
	typedef nTuple<NDIMS, Real> coordinates_type;

	static constexpr Real g_t[NDIMS][NDIMS] = {

	1, 0, 0,

	0, 1, 0,

	0, 0, 1

	};

	template<int I, int J, typename TI>
	constexpr Real g(TI const &) const
	{
		return g_t[I][J];
	}

	//! determind of metric tensor gd=|g|
	template<typename TI>
	constexpr Real gd(TI const &) const
	{
		return 1.0;
	}
	template<typename TI>
	Real g_(TI const & s, unsigned int a, unsigned int b) const
	{
		return g_t[a][b];
	}

	//! diagonal term of metric tensor
	template<typename TI>
	constexpr Real v(TI const & s) const
	{
		return 1.0;
	}
	//! diagonal term of metric tensor
	template<typename TI>
	constexpr Real l_v(TI const & s) const
	{
		return 1.0;
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

	static constexpr int NUM_OF_COMPONENT_TYPE = NDIMS + 1;
	typedef typename OcForest::index_type index_type;

	RectMesh();
	~RectMesh();

	template<typename TDict>
	RectMesh(TDict const & dict)
			: OcForest(dict), Metric(dict)
	{
		Load(dict);
	}

	this_type & operator=(const this_type&) = delete;

	template<typename TDict>
	void Load(TDict const & dict)
	{
	}

	std::ostream & Save(std::ostream &os) const
	{
		return os;
	}

	void Update()
	{
	}
	;

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}

	//***************************************************************************************************
	//*	Miscellaneous
	//***************************************************************************************************
	typedef Real scalar_type;
	//***************************************************************************************************
	//* Media Tags
	//***************************************************************************************************

private:

	MediaTag<this_type> tags_;
public:

	typedef typename MediaTag<this_type>::tag_type tag_type;
	MediaTag<this_type> & tags()
	{
		return tags_;
	}
	MediaTag<this_type> const& tags() const
	{

		return tags_;
	}

	//***************************************************************************************************
	//* Time
	//***************************************************************************************************

private:
	Real dt_ = 0.0; //!< time step
	Real time_ = 0.0;
public:

	void NextTimeStep()
	{
		time_ += dt_;
	}
	Real GetTime() const
	{
		return time_;
	}

	void GetTime(Real t)
	{
		time_ = t;
	}
	inline Real GetDt() const
	{
		CheckCourant();
		return dt_;
	}

	inline void SetDt(Real dt = 0.0)
	{
		dt_ = dt;
		Update();
	}
	double CheckCourant() const
	{
		DEFINE_GLOBAL_PHYSICAL_CONST

		nTuple<3, Real> inv_dx_;
		inv_dx_ = 1.0 / GetDx() / (xmax_ - xmin_);

		Real res = 0.0;

		for (int i = 0; i < 3; ++i)
			res += inv_dx_[i] * inv_dx_[i];

		return std::sqrt(res) * speed_of_light * dt_;
	}

	void FixCourant(Real a)
	{
		dt_ *= a / CheckCourant();
	}

	//***************************************************************************************************
	//* Container: storage depend
	//***************************************************************************************************

	template<typename TV> using Container=std::shared_ptr<TV>;

	nTuple<NUM_OF_DIMS, size_type> strides_ = { 0, 0, 0 };

	inline nTuple<NUM_OF_DIMS, size_type> const & GetStrides() const
	{
		return strides_;
	}

	inline size_type GetNumOfElements(int iform) const
	{

		return (1UL);
	}

	template<int iform, typename TV> inline Container<TV> MakeContainer() const
	{
		return (MEMPOOL.allocate_shared_ptr < TV > (GetNumOfElements(iform)));
	}

	size_type HashIndex( index_type const & s)const
	{
		return OcForest::HashIndex(s,strides_);
	}
	template<typename TI>
	TI HashIndex( TI const & s)const
	{
		return s;
	}
//***************************************************************************************************
// Geometric properties
// Metric
//***************************************************************************************************

	typedef nTuple<3, Real> coordinates_type;

	coordinates_type xmin_ =
	{	0, 0, 0};

	coordinates_type xmax_ =
	{	1, 1, 1};

	coordinates_type inv_L =
	{	1.0, 1.0, 1.0};

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

	struct iterator
	{

		this_type const & mesh;
		container_type data_;
		index_type s_;
		iterator(mesh_type const & m, container_type d, index_type s)
		: mesh(m), data_(d), s_(s)
		{

		}
		iterator(mesh_type const & m, container_type d)
		: mesh(m), data_(d)
		{

		}
		iterator(mesh_type const & m)
		: mesh(m)
		{
		}
		~iterator()
		{

		}
		bool operator==(iterator const & rhs) const
		{
			return (data_ == rhs.data_) && s_ == rhs.s_;
		}
		value_type & operator*()
		{
			return mesh.get_value(data_, s_);
		}
		value_type const& operator*() const
		{
			return mesh.get_value(data_, s_);
		}

		this_type & operator ++()
		{
			s_ = mesh.Next(s_);
			return *this;
		}
	};
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
	Field<Geometry<this_type, IL>, TL> const & f, index_type s)->decltype(f[s])
	{
		return ExteriorDerivative_(f, s);
	}
	template<int IL, typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,
	Field<Geometry<this_type, IL>, TL> const & f, index_type s)->decltype(f[s])
	{
		return Codifferential_(f, s);
	}

	template<int IL, typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>,
	nTuple<NDIMS, TR> const & v, Field<Geometry<this_type, IL>, TL> const & f, index_type s)->decltype(f[s])
	{
		return InteriorProduct_(v, f, s);
	}

private:
//! Form<IR> ^ Form<IR> => Form<IR+IL>
	template<int IL, int IR, typename TL, typename TR> inline Real Wedge_(Field<Geometry<this_type, IL>, TL> const &l,
	Field<Geometry<this_type, IR>, TR> const &r, index_type s) const
	{
		return 0;
	}

	template<typename TL, typename TR> inline auto Wedge_(Field<Geometry<this_type, VERTEX>, TL> const &l,
	Field<Geometry<this_type, VERTEX>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		return l[s] * r[s];
	}

	template<typename TL, typename TR> inline auto Wedge_(Field<Geometry<this_type, VERTEX>, TL> const &l,
	Field<Geometry<this_type, EDGE>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = s & (_MA >> (s.H + 1));
		((l[s - X] + l[s + X]) * 0.5 * r[s]);
	}

	template<typename TL, typename TR> inline auto Wedge_(Field<Geometry<this_type, VERTEX>, TL> const &l,
	Field<Geometry<this_type, FACE>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto Y = _R(_I(s)) & (_MA >> (s.H + 1));
		auto Z = _RR(_I(s)) & (_MA >> (s.H + 1));

		return (l[(s - Y) - Z] + l[(s - Y) + Z] + l[(s + Y) - Z] + l[(s + Y) + Z]) * 0.25 * r[s];
	}

	template<typename TL, typename TR> inline auto Wedge_(Field<Geometry<this_type, VERTEX>, TL> const &l,
	Field<Geometry<this_type, VOLUME>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = _MI >> (s.H + 1);
		auto Y = _MJ >> (s.H + 1);
		auto Z = _MK >> (s.H + 1);

		return (

		l[((s - X) - Y) - Z] + l[((s - X) - Y) + Z] + l[((s - X) + Y) - Z] + l[((s - X) + Y) + Z] +

		l[((s + X) - Y) - Z] + l[((s + X) - Y) + Z] + l[((s + X) + Y) - Z] + l[((s + X) + Y) + Z]

		) * 0.125 * r[s];
	}

	template<typename TL, typename TR> inline auto Wedge_(Field<Geometry<this_type, EDGE>, TL> const &l,
	Field<Geometry<this_type, VERTEX>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		return Wedge_(r, l, s);
	}

	template<typename TL, typename TR> inline auto Wedge_(Field<Geometry<this_type, EDGE>, TL> const &l,
	Field<Geometry<this_type, EDGE>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto Y = _R(_I(s)) & (_MA >> (s.H + 1));
		auto Z = _RR(_I(s)) & (_MA >> (s.H + 1));

		return ((l[s - Y] + l[s + Y]) * (l[s - Z] + l[s + Z]) * 0.25);
	}

	template<typename TL, typename TR> inline auto Wedge_(Field<Geometry<this_type, EDGE>, TL> const &l,
	Field<Geometry<this_type, FACE>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = (_MI >> (s.H + 1));
		auto Y = (_MJ >> (s.H + 1));
		auto Z = (_MK >> (s.H + 1));

		return

		(l[(s - Y) - Z] + l[(s - Y) + Z] + l[(s + Y) - Z] + l[(s + Y) + Z]) * (r[s - X] + r[s + X]) * 0.125 +

		(l[(s - Z) - X] + l[(s - Z) + X] + l[(s + Z) - X] + l[(s + Z) + X]) * (r[s - Y] + r[s + Y]) * 0.125 +

		(l[(s - X) - Y] + l[(s - X) + Y] + l[(s + X) - Y] + l[(s + X) + Y]) * (r[s - Z] + r[s + Z]) * 0.125;
	}

	template<typename TL, typename TR> inline auto Wedge_(Field<Geometry<this_type, FACE>, TL> const &l,
	Field<Geometry<this_type, VERTEX>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		return Wedge_(r, l, s);
	}

	template<typename TL, typename TR> inline auto Wedge_(Field<Geometry<this_type, FACE>, TL> const &l,
	Field<Geometry<this_type, EDGE>, TR> const &r, index_type s) const ->decltype(r[s]*l[s])
	{
		auto X = (_MI >> (s.H + 1));
		auto Y = (_MJ >> (s.H + 1));
		auto Z = (_MK >> (s.H + 1));

		return

		(r[(s - Y) - Z] + r[(s - Y) + Z] + r[(s + Y) - Z] + r[(s + Y) + Z]) * (l[s - X] + l[s + X]) * 0.125 +

		(r[(s - Z) - X] + r[(s - Z) + X] + r[(s + Z) - X] + r[(s + Z) + X]) * (l[s - Y] + l[s + Y]) * 0.125 +

		(r[(s - X) - Y] + r[(s - X) + Y] + r[(s + X) - Y] + r[(s + X) + Y]) * (l[s - Z] + l[s + Z]) * 0.125;
	}

//***************************************************************************************************

	template<int IL, typename TL> inline auto HodgeStar_(Field<Geometry<this_type, IL>, TL> const & f,
	index_type s) const->decltype(f[s])
	{
		auto X = (_MI >> (s.H + 1));
		auto Y = (_MJ >> (s.H + 1));
		auto Z = (_MK >> (s.H + 1));
		return

		(

		f[((s + X) - Y) - Z] + f[((s + X) - Y) + Z] + f[((s + X) + Y) - Z] + f[((s + X) + Y) + Z] +

		f[((s - X) - Y) - Z] + f[((s - X) - Y) + Z] + f[((s - X) + Y) - Z] + f[((s - X) + Y) + Z]

		) * 0.125;
	}

//***************************************************************************************************

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
		auto X = (_MI >> (s.H + 1));
		auto Y = (_MJ >> (s.H + 1));
		auto Z = (_MK >> (s.H + 1));

		return

		(f[s + X] - f[s - X]) * 0.5 * v[0] +

		(f[s + Y] - f[s - Y]) * 0.5 * v[1] +

		(f[s + Z] - f[s - Z]) * 0.5 * v[2];
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
		unsigned int D = (_I(s)) & (_MA >> (s.H + 1));

		return (f[s + D] - f[s - D]) * 0.5 * v[n];
	}

};

}
// namespace simpla

#endif /* RECT_MESH_H_ */
