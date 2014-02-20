/*
 * hex_mesh.h
 *
 *  Created on: 2014年2月20日
 *      Author: salmon
 */

#ifndef HEX_MESH_H_
#define HEX_MESH_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <iostream>
#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <limits>

#include "../fetl/fetl.h"
#include "../utilities/log.h"
#include "../utilities/memory_pool.h"
#include "../utilities/type_utilites.h"
#include "../utilities/utilities.h"
#include "../modeling/media_tag.h"

#include "../fetl/fetl.h"
#include "../physics/constants.h"
#include "../physics/physical_constants.h"

namespace simpla
{
/**
 *
 * Topology
 * RectMesh Axis are perpendicular
 */
class TopologyRect
{
public:
	typedef TopologyRect this_type;

	static constexpr int MAX_NUM_NEIGHBOUR_ELEMENT = 12;
	static constexpr int MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr int NUM_OF_DIMS = 3;
	static constexpr int NUM_OF_COMPONENT_TYPE = NUM_OF_DIMS + 1;

	typedef nTuple<3, Real> coordinates_type;
	typedef Real scalar_type;
	// Topology
	unsigned int DEFAULT_GHOST_WIDTH = 2;

	typedef signed long index_type;

	nTuple<NUM_OF_DIMS, size_t> dims_ = { 0, 0, 0 }; //!< number of cells

	nTuple<NUM_OF_DIMS, size_t> ghost_width_ = { 0, 0, 0 };

	nTuple<NUM_OF_DIMS, size_t> strides_ = { 0, 0, 0 };

	size_t num_cells_ = 0;

	size_t num_grid_points_ = 0;

	// Geometry

	coordinates_type dx_ = { 0, 0, 0 };
	coordinates_type inv_dx_ = { 0, 0, 0 };

	const int num_comps_per_cell_[NUM_OF_COMPONENT_TYPE] = { 1, 3, 3, 1 };

	coordinates_type coordinates_shift_[NUM_OF_COMPONENT_TYPE][NUM_OF_DIMS];

	TopologyRect();

	~TopologyRect();

	this_type & operator=(const this_type&) = delete;

	//***************************************************************************************************
	//* Configure
	//***************************************************************************************************

	template<typename TDict> void Load(TDict const &dict);

	std::ostream & Save(std::ostream &vm) const;

	void Update();

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}

	static inline std::string GetTypeName()
	{
		return "RectMesh";
	}

	inline std::string GetTopologyTypeAsString() const
	{
		return ToString(GetRealNumDimension()) + "DRectMesh";
	}

	// Properties of UniformRectMesh --------------------------------------

	MediaTag<this_type> tags_;

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
	//* Constants
	//***************************************************************************************************
	PhysicalConstants constants_;	//!< Unit System and phyical constants
	PhysicalConstants & constants()
	{
		return constants_;
	}
	PhysicalConstants const& constants() const
	{

		return constants_;
	}

	Real time_, dt_;
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
	bool CheckCourant() const
	{
		DEFINE_PHYSICAL_CONST(constants_);

		Real res = 0.0;

		for (int i = 0; i < 3; ++i)
			res += inv_dx_[i] * inv_dx_[i];
		res = std::sqrt(res) * speed_of_light * dt_;

		if (res > 1.0)
			VERBOSE << "dx/dt > c, Courant condition is violated! ";

		return res < 1.0;
	}

	void FixCourant(Real a = 0.5)
	{
		DEFINE_PHYSICAL_CONST(constants_);

		Real res = 0.0;

		for (int i = 0; i < 3; ++i)
			res += inv_dx_[i] * inv_dx_[i];

		if (std::sqrt(res) * speed_of_light * dt_ > 1.0)
			dt_ = a / (std::sqrt(res) * speed_of_light);

	}

	template<int IN, typename T>
	inline void SetExtent(nTuple<IN, T> const & pmin, nTuple<IN, T> const & pmax)
	{
		int n = IN < NUM_OF_DIMS ? IN : NUM_OF_DIMS;

	}

	template<int IN, typename T>
	inline void SetDimension(nTuple<IN, T> const & d)
	{
		int n = IN < NUM_OF_DIMS ? IN : NUM_OF_DIMS;

		for (int i = 0; i < n; ++i)
		{
			dims_[i] = d[i];
		}

		for (int i = n; i < NUM_OF_DIMS; ++i)
		{
			dims_[i] = 1;
		}
	}

	nTuple<NUM_OF_DIMS, size_t> const & GetDimension() const
	{
		return dims_;
	}

	int GetRealNumDimension() const
	{
		int n = 0;
		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			if (dims_[i] > 1)
				++n;
		}
		return n;
	}
	inline std::vector<size_t> GetShape(int IFORM) const
	{
		std::vector<size_t> res;
		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			if (dims_[i] > 1)
				res.push_back(dims_[i]);
		}
		if (num_comps_per_cell_[IFORM] > 1)
		{
			res.push_back(num_comps_per_cell_[IFORM]);
		}

		return std::move(res);
	}
	inline nTuple<NUM_OF_DIMS, size_t> const & GetStrides() const
	{
		return strides_;
	}

	// General Property -----------------------------------------------

	inline index_type GetNumOfElements(int iform) const
	{

		return (num_grid_points_ * num_comps_per_cell_[iform]);
	}
	template<int IFORM>
	inline int GetNumCompsPerCell() const
	{
		return (num_comps_per_cell_[IFORM]);

	}
	inline index_type GetNumOfVertices(...) const
	{

		return (num_grid_points_);
	}

	//***************************************************************************************************
	//* Container
	//***************************************************************************************************

	template<typename TV> using Container=std::shared_ptr<TV>;

	template<int iform, typename TV> inline Container<TV> MakeContainer() const
	{
		return (MEMPOOL.allocate_shared_ptr<TV>(GetNumOfElements(iform)));
	}

	//***************************************************************************************************
	//* Mesh operation
	//***************************************************************************************************
public:

	//****************************************************************************************************
	//* Index operation
	//*
	//***************************************************************************************************
	/**
	 * 	Thanks my wife Dr. CHEN Xiang Lan, for her advice on  these bitwise operation
	 * 	   n~=3         m            m             m
	 * 	|--------|------------|--------------|-------------|
	 * 	   N         I              J             K
	 *
	 * 	 n+m*3=digits(unsigned long)=63
	 * 	 n=3
	 * 	 m=20
	 */

	typedef signed long shift_type;

	static constexpr int DIGITS_LONG=std::numeric_limits<unsigned long>::digits;//!< signed long is 63bit, unsigned long is 64 bit, add a sign bit

	static constexpr int DIGITS_SHORT=std::numeric_limits<unsigned short>::digits;

	static constexpr int DIGITS_FULL=std::numeric_limits<unsigned long>::digits;

	static constexpr int DIGITS_INDEX=(std::numeric_limits<unsigned long>::digits-3)/3;//!< signed long is 63bit, unsigned long is 64 bit, add a sign bit

	static constexpr int DIGITS_COMP= std::numeric_limits<unsigned long>::digits-DIGITS_INDEX*3;

#define _shift_bit(m) \
			 static_cast<shift_type>((static_cast<unsigned long>((-1L) << (DIGITS_FULL - DIGITS_INDEX)) >> (DIGITS_FULL - DIGITS_INDEX*(m+1) )))
	enum
	{
		IX = 1L, // 0000 0000 0001
		DX= _shift_bit(0),// 0000 0000 1111
		IY = 1L<<DIGITS_SHORT,// 0000 0001 0000
		DY=_shift_bit(1),// 0000 1111 0000
		IZ = 1L<<(DIGITS_SHORT*2),// 0001 0000 0000
		DZ=_shift_bit(2)// 1111 0000 0000

	};

	inline index_type INC(int m) const
	{
		return 1L << ((m % 3) * DIGITS_INDEX);
	}
	inline index_type DES(int m) const
	{
		return _shift_bit((m%3));}

#undef _shift_bit

	template<int M>
	size_t UnpackIndex(index_type s)const
	{
		return (s << (DIGITS_FULL - DIGITS_INDEX*(3-M))) >> (DIGITS_FULL - DIGITS_INDEX );
	}
	template<int M>
	index_type PackIndex(size_t s)const
	{
		return s << ( DIGITS_INDEX* (2-M));
	}

	size_t GetX(index_type s)const
	{
		return UnpackIndex<0>(s);
	}
	size_t GetY(index_type s)const
	{
		return UnpackIndex<1>(s);
	}
	size_t GetZ(index_type s)const
	{
		return UnpackIndex<2>(s);
	}

	size_t GetM(index_type s)const
	{
		return s >> (DIGITS_FULL - DIGITS_COMP );
	}

	template<typename TS>
	index_type PutX(TS s)const
	{
		return PackIndex<0>(static_cast<size_t>(s));
	}

	template<typename TS>
	index_type PutY(TS s)const
	{
		return PackIndex<1>(static_cast<size_t>(s));
	}

	template<typename TS>
	index_type PutZ(TS s)const
	{
		return PackIndex<2>(static_cast<size_t>(s));
	}

	index_type PutM(size_t s)const
	{
		return s << (DIGITS_FULL - DIGITS_COMP );
	}

	// NOTE backward Compatible with old version
	index_type GetComponentIndex(int IFORM,int m,size_t i,size_t j,size_t k)const
	{
		return PutM(m)+PutX(i)+PutX(j)+PutX(k);
	}
	index_type GetComponentIndex( int m,size_t i,size_t j,size_t k)const
	{
		return PutM(m)+PutX(i)+PutX(j)+PutX(k);
	}
	inline index_type Shift(shift_type d, index_type s) const
	{
//FIXME Cycle boundary

		return s&d;
	}

	template<int IFORM>
	inline size_t GetArrayIndex(index_type s) const
	{
		return

		( GetX(s) * strides_[0]+
		GetY(s)* strides_[1]+
		GetZ(s) * strides_[2] )*num_comps_per_cell_[IFORM]+GetM(s);
	}

	template<int IFORM,typename ...TI>
	inline size_t GetArrayIndex(TI ... s) const
	{
		return GetArrayIndex(GetComponentIndex(IFORM,s...));
	}

	//**************************************************************************************************
	inline coordinates_type GetCoordinates(int IFORM, index_type s) const
	{

		coordinates_type res=
		{

			(GetX(s) + coordinates_shift_[IFORM][GetM(s)][0])* dx_[0],

			(GetY(s) + coordinates_shift_[IFORM][GetM(s)][1])* dx_[1],

			(GetZ(s) + coordinates_shift_[IFORM][GetM(s)][2])* dx_[2]
		};

		return std::move(res);
	}
	inline coordinates_type GetCoordinates(int IFORM, int m, size_t i, size_t j, size_t k) const
	{

		coordinates_type res=
		{
			(i+ coordinates_shift_[IFORM][m][0])* dx_[0],

			(j+ coordinates_shift_[IFORM][m][1])* dx_[1],

			(k+ coordinates_shift_[IFORM][m][2])* dx_[2]
		};
		return std::move(res);
	}
	inline size_t GetSubComponent(index_type s) const
	{
		return GetM(s);
	}

	inline index_type GetCellIndex(index_type s) const
	{
		UNIMPLEMENT;
		return s & (1L << (DIGITS_FULL - DIGITS_COMP));
	}
	template<int m>
	inline index_type CycleComp(index_type s) const
	{
		return GetCellIndex(s) + PutM((GetM(s) + m) % 3);
	}
	template<int IFORM, typename TV>
	TV GetWeightOnElement(TV const & v, index_type const &s) const
	{
		return v;
	}

	template<int IFORM, typename TV>
	TV GetWeightOnElement(nTuple<3, TV> const & v, index_type const &s) const
	{
		return v[s % num_comps_per_cell_[IFORM]];
	}

	inline index_type GetNearestVertex(coordinates_type const &x) const
	{
		index_type s = 0;

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			s += static_cast<index_type>(std::floor(x[i] / dx_[i])) * strides_[i];
		}
		return s;
	}

	template<int I>
	inline int GetAdjacentCells(Int2Type<I>, Int2Type<I>, index_type s, index_type *v) const
	{
		if (v != nullptr)
		v[0] = s;
		return 1;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VERTEX>, index_type s, index_type *v) const
	{
		if (v != nullptr)
		{
			v[0] = GetCellIndex(s);
			v[1] = Shift(INC(GetM(s)), v[0]);
		}
		return 2;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<VERTEX>, index_type s, index_type *v) const
	{
		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   2---------------*
		 *        |  /|              /|
		 *          / |             / |
		 *         /  |            /  |
		 *        3---|-----------*   |
		 *        | m |           |   |
		 *        |   1-----------|---*
		 *        |  /            |  /
		 *        | /             | /
		 *        |/              |/
		 *        0---------------*---> x
		 *
		 *
		 */

		if (v != nullptr)
		{
			v[0] = GetCellIndex(s);
			v[1] = Shift(INC(GetM(s) + 1), s);
			v[2] = Shift(INC(GetM(s) + 1) | INC(GetM(s) + 2), s);
			v[3] = Shift(INC(GetM(s) + 2), s);
		}
		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<VERTEX>, index_type s, index_type *v) const
	{
		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *          / |             / |
		 *         /  |            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        |   2-----------|---3
		 *        |  /            |  /
		 *        | /             | /
		 *        |/              |/
		 *        0---------------1   ---> x
		 *
		 *
		 */

		if (v != nullptr)
		{
			v[0] = GetCellIndex(s);
			v[1] = Shift(INC(0), s);
			v[2] = Shift(INC(1) | INC(1), s);
			v[3] = Shift(INC(1), s);

			v[4] = Shift(INC(2), s);
			v[5] = Shift(INC(2) | INC(0), s);
			v[6] = Shift(INC(2) | INC(1) | INC(1), s);
			v[7] = Shift(INC(2) | INC(1), s);
		}
		return 8;
	}

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<EDGE>, index_type s, index_type *v) const
	{
		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *          2 |             / |
		 *         /  1            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        |   2-----------|---3
		 *        3  /            |  /
		 *        | 0             | /
		 *        |/              |/
		 *        0------E0-------1   ---> x
		 *
		 *
		 */

		if (v != nullptr)
		{
			v[0] = s + PutM(0);
			v[1] = s + PutM(1);
			v[2] = s + PutM(2);
			v[3] = Shift(DES(1), s) + PutM(0);
			v[4] = Shift(DES(2), s) + PutM(1);
			v[5] = Shift(DES(2), s) + PutM(2);
		}
		return 6;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<EDGE>, index_type s, index_type *v) const
	{

		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *          2 |             / |
		 *         /  1            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        |   2-----------|---3
		 *        3  /            |  /
		 *        | 0             | /
		 *        |/              |/
		 *        0---------------1   ---> x
		 *
		 *
		 */

		if (v != nullptr)
		{
			v[0] = CycleComp<1>(s);
			v[1] = CycleComp<2>(s);
			v[2] = CycleComp<1>(Shift(INC(GetM(s) + 1), s));
			v[2] = CycleComp<2>(Shift(INC(GetM(s) + 2), s));
		}
		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<EDGE>, index_type s, index_type *v) const
	{

		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6------10-------7
		 *        |  /|              /|
		 *         11 |             9 |
		 *         /  7            /  6
		 *        4---|---8-------5   |
		 *        |   |           |   |
		 *        |   2-------2---|---3
		 *        4  /            5  /
		 *        | 3             | 1
		 *        |/              |/
		 *        0-------0-------1   ---> x
		 *
		 *
		 */

		if (v != nullptr)
		{
			v[0] = s + PutM(0);
			v[1] = Shift(INC(0), s) + PutM(1);
			v[2] = Shift(INC(1), s) + PutM(0);
			v[3] = s + PutM(1);

			v[4] = s + PutM(2);
			v[5] = Shift(INC(0), s) + PutM(2);
			v[6] = Shift(INC(1) | INC(0), s) + PutM(2);
			v[7] = Shift(INC(1), s) + PutM(2);

			v[8] = Shift(INC(2), s) + PutM(0);
			v[9] = Shift(INC(2) | INC(0), s) + PutM(1);
			v[10] = Shift(INC(2) | INC(1), s) + PutM(0);
			v[11] = Shift(INC(2), s) + PutM(1);

		}
		return 12;
	}

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<FACE>, index_type s, index_type *v) const
	{
		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        | 0 2-----------|---3
		 *        |  /            |  /
		 *   11   | /      8      | /
		 *      3 |/              |/
		 * -------0---------------1   ---> x
		 *       /| 1
		 *10    / |     9
		 *     /  |
		 *      2 |
		 *
		 *
		 *
		 *              |
		 *          7   |   4
		 *              |
		 *      --------*---------
		 *              |
		 *          6   |   5
		 *              |
		 *
		 *
		 */

		if (v != nullptr)
		{
			v[0] = s;
			v[1] = Shift(DES(2), s);
			v[2] = Shift(DES(2) | DES(1), s);
			v[3] = Shift(DES(1), s);

			v[4] = s + PutM(1);
			v[5] = Shift(DES(2), s) + PutM(1);
			v[6] = Shift(DES(0) | DES(2), s) + PutM(1);
			v[7] = Shift(DES(0), s) + PutM(1);

			v[8] = s + PutM(2);
			v[9] = Shift(DES(1), s) + PutM(2);
			v[10] = Shift(DES(1) | DES(0), s) + PutM(2);
			v[11] = Shift(DES(0), s) + PutM(2);

		}
		return 12;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<FACE>, index_type s, index_type *v) const
	{

		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        |   2-----------|---3
		 *        |  /  0         |  /
		 *        | /      1      | /
		 *        |/              |/
		 * -------0---------------1   ---> x
		 *       /|
		 *      / |   3
		 *     /  |       2
		 *        |
		 *
		 *
		 *
		 *              |
		 *          7   |   4
		 *              |
		 *      --------*---------
		 *              |
		 *          6   |   5
		 *              |
		 *
		 *
		 */

		if (v != nullptr)
		{
			v[0] = CycleComp<1>(s);
			v[1] = CycleComp<2>(s);
			v[2] = CycleComp<1>(Shift(DES(GetM(s) + 2), s));
			v[2] = CycleComp<2>(Shift(DES(GetM(s) + 1), s));
		}
		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<FACE>, index_type s, index_type *v) const
	{

		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^    /
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |    5        / |
		 *        |/  |     1      /  |
		 *        4---|-----------5   |
		 *        | 0 |           | 2 |
		 *        |   2-----------|---3
		 *        |  /    3       |  /
		 *        | /       4     | /
		 *        |/              |/
		 * -------0---------------1   ---> x
		 *       /|
		 *
		 */

		if (v != nullptr)
		{
			v[0] = s + PutM(0);
			v[1] = Shift(INC(1), s) + PutM(1);
			v[2] = Shift(INC(0), s) + PutM(0);
			v[3] = s + PutM(1);

			v[4] = s + PutM(2);
			v[5] = Shift(INC(2), s) + PutM(2);

		}
		return 6;
	}

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<VOLUME>, index_type s, index_type *v) const
	{
		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *        4---|-----------5   |
		 *   3    |   |    0      |   |
		 *        |   2-----------|---3
		 *        |  /            |  /
		 *        | /             | /
		 *        |/              |/
		 * -------0---------------1   ---> x
		 *  3    /|       1
		 *      / |
		 *     /  |
		 *        |
		 *
		 *
		 *
		 *              |
		 *          7   |   4
		 *              |
		 *      --------*---------
		 *              |
		 *          6   |   5
		 *              |
		 *
		 *
		 */

		if (v != nullptr)
		{
			v[0] = GetCellIndex(s);
			v[1] = Shift(DES(0), s);
			v[2] = Shift(DES(0) | DES(1), s);
			v[3] = Shift(DES(1), s);

			v[4] = Shift(DES(2), s);
			v[5] = Shift(DES(2) | DES(0), s);
			v[6] = Shift(DES(2) | DES(0) | DES(1), s);
			v[7] = Shift(DES(2) | DES(1), s);

		}
		return 8;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VOLUME>, index_type s, index_type *v) const
	{

		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        |   2-----------|---3
		 *        |  /  0         |  /
		 *        | /      1      | /
		 *        |/              |/
		 * -------0---------------1   ---> x
		 *       /|
		 *      / |   3
		 *     /  |       2
		 *        |
		 *
		 *
		 *
		 *              |
		 *          7   |   4
		 *              |
		 *      --------*---------
		 *              |
		 *          6   |   5
		 *              |
		 *
		 *
		 */

		if (v != nullptr)
		{
			v[0] = GetCellIndex(s);
			v[1] = Shift(DES(GetM(s) + 1), s);
			v[2] = Shift(DES(GetM(s) + 1) | DES(GetM(s) + 2), s);
			v[3] = Shift(DES(GetM(s) + 2), s);
		}
		return 4;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<VOLUME>, index_type s, index_type *v) const
	{

		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^    /
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *        4---|-----------5   |
		 *        | 0 |           |   |
		 *        |   2-----------|---3
		 *        |  /            |  /
		 *        | /             | /
		 *        |/              |/
		 * -------0---------------1   ---> x
		 *       /|
		 *
		 */

		if (v != nullptr)
		{
			v[0] = GetCellIndex(s);
			v[1] = Shift(DES(GetM(s)), s);

		}
		return 2;
	}

	//***************************************************************************************************
	//  Traversal
	//
	//***************************************************************************************************

	template<typename ... Args>
	void Traversal(Args const &...args) const
	{
		ParallelTraversal(std::forward<Args const &>(args)...);
	}

	template<typename ...Args> void ParallelTraversal(Args const &...args) const;

	template<typename ...Args> void SerialTraversal(Args const &...args) const;

	void _Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
	std::function<void(index_type)> const &funs) const;

	template<typename Fun, typename TF, typename ... Args> inline
	void SerialForEach(Fun const &fun, TF const & l, Args const& ... args) const
	{
		SerialTraversal(FieldTraits<TF>::IForm, [&]( index_type s)
		{	fun(get(l,s),get(args,s)...);});
	}

	template<typename Fun, typename TF, typename ... Args> inline
	void SerialForEach(Fun const &fun, TF *l, Args const& ... args) const
	{
		if (l == nullptr)
		ERROR << "Access value to an uninitilized container!";

		SerialTraversal(FieldTraits<TF>::IForm, [&]( index_type s)
		{	fun(get(l,s),get(args,s)...);});
	}

	template<typename Fun, typename TF, typename ... Args> inline
	void ParallelForEach(Fun const &fun, TF const & l, Args const& ... args) const
	{
		ParallelTraversal(FieldTraits<TF>::IForm, [&]( index_type s)
		{	fun(get(l,s),get(args,s)...);});
	}

	template<typename Fun, typename TF, typename ... Args> inline
	void ParallelForEach(Fun const &fun, TF *l, Args const& ... args) const
	{
		if (l == nullptr)
		ERROR << "Access value to an uninitilized container!";

		ParallelTraversal(FieldTraits<TF>::IForm, [&]( index_type s)
		{	fun(get(l,s),get(args,s)...);});
	}

	template<typename Fun, typename TF, typename ... Args> inline
	void ForEach(Fun const &fun, TF const & l, Args const& ... args) const
	{
		ParallelForEach(fun, l, std::forward<Args const &>(args)...);
	}
	template<typename Fun, typename TF, typename ... Args> inline
	void ForEach(Fun const &fun, TF *l, Args const& ... args) const
	{
		ParallelForEach(fun, l, std::forward<Args const &>(args)...);
	}

	//***************************************************************************************************
	//* Container/Field operation
	//* Field vs. Mesh
	//***************************************************************************************************

	template<typename TV>
	void SetFieldValue(Field<Geometry<this_type, EDGE>, TV> * f, nTuple<3, TV> const &v, index_type s) const
	{
		get(f,s+PutM(0)) = v[0];
		get(f,s+PutM(1)) = v[1];
		get(f,s+PutM(2)) = v[2];
	}

	template<typename TV>
	void SetFieldValue(Field<Geometry<this_type, FACE>, TV> * f, nTuple<3, TV> const &v, index_type s) const
	{
		get(f,s+PutM(0)) = v[0];
		get(f,s+PutM(1)) = v[1];
		get(f,s+PutM(2)) = v[2];
	}

	template<typename TV>
	void SetFieldValue(Field<Geometry<this_type, VERTEX>, TV> * f, TV const &v, index_type s) const
	{
		get(f,s ) = v;
	}

	template<typename TV>
	void SetFieldValue(Field<Geometry<this_type, VOLUME>, TV> * f, TV const &v, index_type s) const
	{
		get(f,s ) = v;
	}

	template<typename TL, typename TR> void AssignContainer(int IFORM, TL * lhs, TR const &rhs) const
	{
		ParallelTraversal(IFORM, [&]( index_type s)
		{	get(lhs,s)=get(rhs,s);});

	}

	template<typename TL, int IL>
	typename std::enable_if<IL == 1 | IL == 2, void>::type AssignContainer(Field<Geometry<this_type, IL>, TL> * lhs,
	typename Field<Geometry<this_type, IL>, TL>::field_value_type const &rhs) const
	{
		ParallelTraversal(0, [&]( index_type s)
		{
			get(lhs,s+PutM(0))=rhs[0];
			get(lhs,s+PutM(1))=rhs[1];
			get(lhs,s+PutM(2))=rhs[2];

		});
	}

	template<typename TL, int IL>
	typename std::enable_if<IL == 0 | IL == 3, void>::type AssignContainer(Field<Geometry<this_type, IL>, TL> * lhs,
	typename Field<Geometry<this_type, IL>, TL>::field_value_type const &rhs) const
	{
		ParallelTraversal(0, [&]( index_type s)
		{	get(lhs,s)=rhs;});
	}

	template<typename T>
	inline typename std::enable_if<!is_field<T>::value, T>::type get(T const &l, index_type) const
	{
		return std::move(l);
	}

	template<int IFORM, typename TL> inline typename Field<Geometry<this_type, IFORM>, TL>::value_type & get(
	Field<Geometry<this_type, IFORM>, TL> *l, index_type s) const
	{
		return l->get( s );
	}

	template<int IFORM, typename TL>
	typename Field<Geometry<this_type, IFORM>, TL>::value_type const &get(Field<Geometry<this_type, IFORM>, TL> const & l,
	index_type s) const
	{
		return (l.get( s ));
	}

	template<int IFORM, int TOP, typename TL>
	typename Field<Geometry<this_type, IFORM>, UniOp<TOP, TL> >::value_type get(
	Field<Geometry<this_type, IFORM>, UniOp<TOP, TL> > const & l, index_type s) const
	{
		return (l.get( s ));
	}

	template<int IFORM, int TOP, typename TL, typename TR>
	typename Field<Geometry<this_type, IFORM>, BiOp<TOP, TL, TR> >::value_type get(
	Field<Geometry<this_type, IFORM>, BiOp<TOP, TL, TR> > const & l, index_type s) const
	{
		return (l.get( s ));
	}

	template<typename TV>
	TV & get_value(Container<TV> & d, size_t s)const
	{
		return *(d.get() + s );
	}
	template<typename TV>
	TV const & get_value(Container<TV> const& d, size_t s) const
	{
		return *(d.get() + s );
	}

	template<typename TFUN>
	inline void TraversalSubComponent(int IFORM, index_type s, TFUN const & fun) const
	{
		int num = num_comps_per_cell_[IFORM];
		for (int i = 0; i < num; ++i)
		{
			fun(s * num + i);
		}
	}

	//***************************************************************************************************
	// Particle  in Cell
	// Begin
	//***************************************************************************************************

	/**
	 * Locate the cell containing a specified point.
	 * @param x
	 * @param r local parameter coordinates
	 * @return index of cell
	 */
	inline index_type SearchCell(coordinates_type const &x, Real * r = nullptr) const
	{

		double i[NUM_OF_DIMS] =
		{	0, 0, 0};

		for (int n = 0; n < NUM_OF_DIMS; ++n)
		{

			double e = std::modf(x[n] * inv_dx_[n], &(i[n]));

			if (r != nullptr)
			{
				r[n] = e;
			}

		}

		return PutX ( i[0] ) + PutY ( i[1] ) + PutZ ( i[2] );
	}

	/**
	 *  Speed up version SearchCell, for curvline or unstructured grid
	 * @param
	 * @param x
	 * @param pcoords
	 * @return index of cell
	 */
	inline index_type SearchCell(index_type const &hint_idx, coordinates_type const &x, Real *pcoords = nullptr) const
	{
		return SearchCell(x, pcoords);
	}

	inline index_type SearchCellFix(index_type s, Real *r) const
	{
		UNIMPLEMENT;

		index_type d = 0;
		for (int i = 0; i < 3; ++i)
		{
			if (dims_[i] <= 1)
			{
				r[i] = 0;
			}
			else if (r[i] < 0)
			{
				double n;
				r[i] = std::modf(r[i], &n) + 1;
				d |= DES(i) * static_cast<signed int>(n - 1);
			}
			else if (r[i] > 1.0)
			{
				double n;
				r[i] = std::modf(r[i], &n);
				d |= INC(i) * static_cast<signed int>(n);
			}

		}

		return Shift(d, s);

	}

	/**
	 *
	 * @param s
	 * @param x
	 * @return number vertex
	 * 	     2 for uniform rectangle
	 * 	     4 for Tetrahedron
	 */
	inline int GetCellShape(index_type s, coordinates_type * x = nullptr) const
	{
		if (x != nullptr)
		{
			x[0] = GetCoordinates(0, s);
			x[1] = x[0] + dx_;
		}
		return 2;
	}

	index_type Refelect(index_type hint_s, Real dt, coordinates_type * x, nTuple<3, Real> * v) const
	{
//		coordinates_type r;
//		r=*x;
//		index_type s = SearchCell(hint_s,&(r)[0]);
//
//		shift_type d=0;
//
//		for(int i=0;i<3;++i)
//		{
//			auto a=r[i]-dt*(*v)[i]*inv_dx_[i];
//			if(a <0)
//			{
//				d|=DES(i);
//			}
//			else if(a >1 )
//			{
//				d|=INC(i);
//			}
//			else
//			{
//				continue;
//			}
//			v[i] *=-1;
//			r[i] =1.0-(*x)[i];
//		}
//
//		if(d!=0)
//		{
//			*x=GetGlobalCoordinates(s,r);
//			s= Shift(d,s);
//		}
		return 0;

	}
	//***************************************************************************************************
	// Particle <=> Mesh/Cache
	// Begin

	//  Scatter/Gather to Cache
	template<int I> inline index_type GetAffectedPoints(Int2Type<I>, index_type const & s = 0,
	index_type * points = nullptr, int affect_region = 2) const
	{
		size_t i, j, k;

		size_t num = num_comps_per_cell_[I];

		if (points != nullptr)
		{
			int t = 0;

			index_type i_b = (dims_[0] > 1) ? i - affect_region + 1 : 0;
			index_type i_e = (dims_[0] > 1) ? i + affect_region + 1 : 1;
			index_type j_b = (dims_[1] > 1) ? j - affect_region + 1 : 0;
			index_type j_e = (dims_[1] > 1) ? j + affect_region + 1 : 1;
			index_type k_b = (dims_[2] > 1) ? k - affect_region + 1 : 0;
			index_type k_e = (dims_[2] > 1) ? k + affect_region + 1 : 1;

			for (index_type i = i_b; i < i_e; i += IX)
			for (index_type j = j_b; j < j_e; j += IY)
			for (index_type k = k_b; k < k_e; k += IZ)
			{
				points[t] = i + j + k;

				for (int m = 1; m < num; ++m)
				{
					points[t + m] = points[t] + PutM(m);
				}
				t += num;
			}
		}

		size_t w = 1;

		for (int i = 0; i < 3; ++i)
		{
			if (dims_[i] > 1)
			{
				w *= (affect_region * 2);
			}
		}
		return w * num;
	}

private:
	inline index_type GetCacheCoordinates(int w,index_type *sx ,Real *r )const
	{

		sx[0]= (dims_[0]<=1)?0:((dims_[1]<=1)?1:w*2) *((dims_[2]<=1)?1:w*2) ,

		sx[1]= (dims_[1]<=1)?0:((dims_[2]<=1)?1:w*2);

		sx[2]= (dims_[2]<=1)?0:1;

		///@NOTE Dot not check boundary, user should ensure abs(r)<w

		index_type res=0;

		for(int n=0;n<3;++n)
		{
			if(dims_[n]>1)
			{
				Real i;

				r[n]=std::modf(r[n],&i);

				res+=static_cast<index_type>(i)*sx[n];
			}
			else
			{
				r[n]=0;
			}

		}

		return res;
	}

#define DEF_INTERPOLATION_SCHEME(_LEFT_,_RIGHT_)                                                       \
	_LEFT_ cache[(o)*num_of_comp+comp_num] _RIGHT_ * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);       \
	_LEFT_ cache[(o+sx[0])*num_of_comp+comp_num] _RIGHT_ * r[0] * (1.0 - r[1]) * (1.0 - r[2]);         \
	_LEFT_ cache[(o+sx[1])*num_of_comp+comp_num] _RIGHT_ * (1.0 - r[0]) * r[1]* (1.0 - r[2]);          \
	_LEFT_ cache[(o+sx[0]+sx[1])*num_of_comp+comp_num] _RIGHT_ * r[0] * r[1]* (1.0 - r[2]);            \
	_LEFT_ cache[(o+sx[2])*num_of_comp+comp_num] _RIGHT_ * (1.0 - r[0]) * (1.0 - r[1]) * r[2];         \
	_LEFT_ cache[(o+sx[0]+sx[2])*num_of_comp+comp_num] _RIGHT_ * r[0] * (1.0 - r[1]) * r[2];           \
	_LEFT_ cache[(o+sx[1]+sx[2])*num_of_comp+comp_num] _RIGHT_ * (1.0 - r[0]) * r[1]* r[2];            \
	_LEFT_ cache[(o+sx[0]+sx[1]+sx[2])*num_of_comp+comp_num] _RIGHT_ * r[0] * r[1]* r[2];

	template<typename TV, typename TW>
	inline void ScatterToCache(Real const x[3], TW const &v, TV *cache, int w, int num_of_comp = 1, int comp_num = 0) const
	{
		Real r[3] =
		{	x[0], x[1], x[2]};
		index_type sx[3];
		index_type o = GetCacheCoordinates(w, sx, r);

		DEF_INTERPOLATION_SCHEME(, +=v)
	}

	template<typename TV, typename TW>
	inline void GatherFromCache(Real const x[3], TV const*cache, TW *res, int w, int num_of_comp = 1,
	int comp_num = 0) const
	{
		Real r[3] =
		{	x[0], x[1], x[2]};

		index_type sx[3];
		index_type o = GetCacheCoordinates(w, sx, r);

		(*res) = 0;
		DEF_INTERPOLATION_SCHEME((*res)+=,)
	}
#undef DEF_INTERPOLATION_SCHEME

public:

	/**
	 *
	 * @param
	 * @param rr  is cache coordinate not global coordinate
	 * @param v
	 * @param cache
	 * @param w
	 */
	template<typename TV,typename TW>
	inline void Scatter(Int2Type<VERTEX>,Real const *rr, TW const & v,TV* cache, int w = 2) const
	{
		ScatterToCache(rr,std::forward<TW const &>(v),cache,w );
	}

	template<typename TV, typename TW>
	inline void Gather(Int2Type<VERTEX>, Real const *rr, TV const* cache, TW* v, int w = 2) const
	{
		GatherFromCache(rr, cache, v, w);
	}

	template<typename TV, typename TW>
	inline void Scatter(Int2Type<VOLUME>, Real const *rr, TW const & v, TV* cache, int w = 2) const
	{
		Real r[3] =
		{	rr[0] - 0.5, rr[1] - 0.5, rr[2] - 0.5};

		ScatterToCache(r, std::forward<TW const &>(v), cache, w);
	}

	template<typename TV, typename TW>
	inline void Gather(Int2Type<VOLUME>, Real const *rr, TV const* cache, TW* v, int w = 2) const
	{
		Real r[3] =
		{	rr[0] - 0.5, rr[1] - 0.5, rr[2] - 0.5};

		GatherFromCache(r, cache, v, w);
	}

	template<typename TV, typename TW>
	inline void Scatter(Int2Type<EDGE>, Real const *rr, nTuple<3, TW> const & v, TV* cache, int w = 2) const
	{

		for (int m = 0; m < 3; ++m)
		{
			Real r[3] =
			{	rr[0], rr[1], rr[2]};
			r[m] -= 0.5;
			ScatterToCache(r, v[m], cache, w, 3, m);

		}
	}

	template<typename TV, typename TW>
	inline void Gather(Int2Type<EDGE>, Real const *rr, TV const* cache, nTuple<3, TW>* v, int w = 2) const
	{
		(*v) = 0;
		for (int m = 0; m < 3; ++m)
		{
			Real r[3] =
			{	rr[0], rr[1], rr[2]};
			r[m] -= 0.5;
			GatherFromCache(r, cache, &(*v)[m], w, 3, m);
		}
	}

	template<typename TV, typename TW>
	inline void Scatter(Int2Type<FACE>, Real const *rr, nTuple<3, TW> const & v, TV* cache, int w = 2) const
	{

		for (int m = 0; m < 3; ++m)
		{
			Real r[3] =
			{	rr[0], rr[1], rr[2]};
			r[(m + 1) % 2] -= 0.5;
			r[(m + 2) % 2] -= 0.5;
			ScatterToCache(r, v[m], cache, w, 3, m);

		}
	}

	template<typename TV, typename TW>
	inline void Gather(Int2Type<FACE>, Real const *rr, TV const* cache, nTuple<3, TW>* v, int w = 2) const
	{
		(*v) = 0;
		for (int m = 0; m < 3; ++m)
		{
			Real r[3] =
			{	rr[0], rr[1], rr[2]};
			r[(m + 1) % 2] -= 0.5;
			r[(m + 2) % 2] -= 0.5;
			GatherFromCache(r, cache, &(*v)[m], w, 3, m);
		}
	}

	template<int IFORM, typename TV, typename TR> void GetMeanValue(TV const * cache, TR * v, int affect_region) const
	{
		index_type w = 2 * affect_region;

		index_type sx[3];

		sx[0] = (dims_[0] <= 1) ? 0 : ((dims_[1] <= 1) ? 1 : w * 2) * ((dims_[2] <= 1) ? 1 : w * 2),

		sx[1] = (dims_[1] <= 1) ? 0 : ((dims_[2] <= 1) ? 1 : w * 2);

		sx[2] = (dims_[2] <= 1) ? 0 : 1;

		int count = 0;

		for (index_type i = 0, ie = ((dims_[0] > 1) ? w : 1); i < ie; ++i)
		for (index_type j = 0, je = ((dims_[1] > 1) ? w : 1); j < je; ++j)
		for (index_type k = 0, ke = ((dims_[2] > 1) ? w : 1); k < ke; ++k)
		{
			index_type s = i * sx[0] + j * sx[1] + k * sx[2];
			++count;

			for (index_type m = 0; m < num_comps_per_cell_[IFORM]; ++m)
			{
				v[m] += cache[s * num_comps_per_cell_[IFORM] + m];
			}
		}

		for (index_type m = 0; m < num_comps_per_cell_[IFORM]; ++m)
		{
			v[m] /= static_cast<Real>(count);
		}

	}

	//  Scatter/Gather to Mesh

#define DEF_INTERPOLATION_SCHEME(_LEFT_,_RIGHT_)                                                       \
	_LEFT_ cache[(o)*num_of_comp+m] _RIGHT_ * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);       \
	_LEFT_ cache[Shift(IX,o)*num_of_comp+m] _RIGHT_ * r[0] * (1.0 - r[1]) * (1.0 - r[2]);         \
	_LEFT_ cache[Shift(IY,o)*num_of_comp+m] _RIGHT_ * (1.0 - r[0]) * r[1]* (1.0 - r[2]);          \
	_LEFT_ cache[Shift(IX|IY,o)*num_of_comp+m] _RIGHT_ * r[0] * r[1]* (1.0 - r[2]);            \
	_LEFT_ cache[Shift(IZ,o)*num_of_comp+m] _RIGHT_ * (1.0 - r[0]) * (1.0 - r[1]) * r[2];         \
	_LEFT_ cache[Shift(IX|IZ,o)*num_of_comp+m] _RIGHT_ * r[0] * (1.0 - r[1]) * r[2];           \
	_LEFT_ cache[Shift(IY|IZ,o)*num_of_comp+m] _RIGHT_ * (1.0 - r[0]) * r[1]* r[2];            \
	_LEFT_ cache[Shift(IX|IY|IZ,o)*num_of_comp+m] _RIGHT_ * r[0] * r[1]* r[2];

	/**
	 *
	 * @param
	 * @param o
	 * @param r is cell local coordinate not global coordiante
	 * @param m
	 * @param v
	 * @param cache
	 */
	template<typename TV, typename TW>
	inline void Scatter(Int2Type<VERTEX>, index_type o, Real const r[3], TW const &v, TV *cache) const
	{
		int num_of_comp = num_comps_per_cell_[VERTEX];
		int m = 0;

		DEF_INTERPOLATION_SCHEME(, +=v)
	}

	template<typename TV, typename TW>
	inline void Gather(Int2Type<VERTEX>, index_type o, Real const r[3], TV const*cache, TW *v) const
	{
		int num_of_comp = num_comps_per_cell_[VERTEX];
		int m = 0;
		(*v) = 0;
		DEF_INTERPOLATION_SCHEME((*v)+=,)
	}

	template<typename TV, typename TW>
	inline void Scatter(Int2Type<EDGE>, index_type o, Real const rr[3], TW const &v, TV *cache) const
	{
		int num_of_comp = num_comps_per_cell_[FACE];
		for (int m = 0; m < num_of_comp; ++m)
		{
			Real r[3] =
			{	rr[0], rr[1], rr[2]};

			r[m] -= 0.5;
			o = SearchCellFix(o, r);
			DEF_INTERPOLATION_SCHEME(, +=v[m])
		}
	}

	template<typename TV, typename TW>
	inline void Gather(Int2Type<EDGE>, index_type o, Real const rr[3], TV const*cache, TW *v) const
	{
		int num_of_comp = num_comps_per_cell_[FACE];

		for (int m = 0; m < num_of_comp; ++m)
		{
			Real r[3] =
			{	rr[0], rr[1], rr[2]};

			r[m] -= 0.5;
			(*v) = 0;
			o = SearchCellFix(o, r);
			DEF_INTERPOLATION_SCHEME((*v)[m]+=,)
		}
	}

	template<typename TV, typename TW>
	inline void Scatter(Int2Type<FACE>, index_type o, Real const rr[3], TW const &v, TV *cache) const
	{

		int num_of_comp = num_comps_per_cell_[FACE];
		for (int m = 0; m < num_of_comp; ++m)
		{
			Real r[3] =
			{	rr[0], rr[1], rr[2]};

			r[(m + 1) % 3] -= 0.5;
			r[(m + 2) % 3] -= 0.5;
			o = SearchCellFix(o, r);
			DEF_INTERPOLATION_SCHEME(, +=v[m])
		}
	}

	template<typename TV, typename TW>
	inline void Gather(Int2Type<FACE>, index_type o, Real const rr[3], TV const*cache, TW *v) const
	{

		int num_of_comp = num_comps_per_cell_[FACE];

		for (int m = 0; m < num_of_comp; ++m)
		{
			Real r[3] =
			{	rr[0], rr[1], rr[2]};

			r[(m + 1) % 3] -= 0.5;
			r[(m + 2) % 3] -= 0.5;

			(*v) = 0;
			o = SearchCellFix(o, r);
			DEF_INTERPOLATION_SCHEME((*v)[m]+=,)
		}

	}

	template<typename TV, typename TW>
	inline void Scatter(Int2Type<VOLUME>, index_type o, Real const rr[3], TW const &v, TV *cache) const
	{
		Real r[3]
		{	rr[0] - 0.5, rr[1] - 0.5, rr[2] - 0.5};

		int num_of_comp = num_comps_per_cell_[VOLUME];
		int m = 0;
		o = SearchCellFix(o, r);
		DEF_INTERPOLATION_SCHEME(, +=v)
	}

	template<typename TV, typename TW>
	inline void Gather(Int2Type<VOLUME>, index_type o, Real const rr[3], TV const*cache, TW *v) const
	{
		Real r[3]
		{	rr[0] - 0.5, rr[1] - 0.5, rr[2] - 0.5};

		int num_of_comp = num_comps_per_cell_[VOLUME];
		int m = 0;
		(*v) = 0;
		o = SearchCellFix(o, r);
		DEF_INTERPOLATION_SCHEME((*v)+=,)
	}

#undef DEF_INTERPOLATION_SCHEME

	//***************************************************************************************************
	// Mesh vs Mesh

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

	template<int IF, typename T> inline auto//
	mapto(Int2Type<IF>, T const &l, ...) const
	ENABLE_IF_DECL_RET_TYPE(is_primitive<T>::value,l)

	template<int IF, typename TL> inline auto mapto(Int2Type<IF>, Field<Geometry<this_type, IF>, TL> const &l,
	index_type s) const
	DECL_RET_TYPE ((get(l,s)))

	template<typename TL> inline auto mapto(Int2Type<EDGE>, Field<Geometry<this_type, VERTEX>, TL> const &l,
	index_type s) const
	DECL_RET_TYPE( ((get(l,Shift(INC(GetM(s)),s)) + get(l, s))*0.5) )

	template<typename TL> inline auto//
	mapto(Int2Type<FACE>, Field<Geometry<this_type, VERTEX>, TL> const &l, index_type s) const
	DECL_RET_TYPE((
			(
					get(l, s)+
					get(l,Shift(INC(GetM(s)+1),s))+
					get(l,Shift(INC(GetM(s)+2),s))+
					get(l,Shift(INC(GetM(s)+1) | INC(GetM(s)+2) ,s))
			)*0.25

	))
	template<typename TL> inline auto//
	mapto(Int2Type<VOLUME>, Field<Geometry<this_type, VERTEX>, TL> const &l, index_type s) const
	DECL_RET_TYPE(( (
					get(l ,s)+
					get(l,Shift(IX,s))+
					get(l,Shift(IY,s))+
					get(l,Shift(IZ,s))+

					get(l,Shift(IX|IY,s))+
					get(l,Shift(IZ|IX,s))+
					get(l,Shift(IZ|IY,s))+
					get(l,Shift(IZ|IX|IY,s))

			)*0.125

	))

	template<typename TL>
	inline auto mapto(Int2Type<VERTEX>, Field<Geometry<this_type, EDGE>, TL> const &l, index_type s) const
	DECL_RET_TYPE( (get(l, s)+get(l, Shift(DES(GetM(s)),s)))*0.5 )

	template<typename TL>
	inline auto mapto(Int2Type<FACE>, Field<Geometry<this_type, EDGE>, TL> const &l, index_type s) const
	DECL_RET_TYPE( (get(l, s)+
			get(l, Shift(INC(GetM(s)+1),s))+
			get(l, Shift(INC(GetM(s)+2),s))+
			get(l, Shift(INC(GetM(s)+1)|INC(GetM(s)+2),s))+

			get(l, Shift(DES(GetM(s)),s))+
			get(l, Shift(DES(GetM(s))|INC(GetM(s)+1),s))+
			get(l, Shift(DES(GetM(s))|INC(GetM(s)+2),s))+
			get(l, Shift(DES(GetM(s))|INC(GetM(s)+1)|INC(GetM(s)+2),s))
	)*0.125 )

	template<typename TL>
	inline auto mapto(Int2Type<VOLUME>, Field<Geometry<this_type, EDGE>, TL> const &l, index_type s) const
	DECL_RET_TYPE( (get(l, s)+
			get(l, Shift(INC(GetM(s)+1),s))+
			get(l, Shift(INC(GetM(s)+2),s))+
			get(l, Shift(INC(GetM(s)+1)|INC(GetM(s)+2),s))
	)*0.25 )

	template<typename TL>
	inline auto mapto(Int2Type<VERTEX>, Field<Geometry<this_type, FACE>, TL> const &l, index_type s) const
	DECL_RET_TYPE( (get(l, s)+
			get(l, Shift(DES(GetM(s)+1),s))+
			get(l, Shift(DES(GetM(s)+2),s))+
			get(l, Shift(DES(GetM(s)+1)|DES(GetM(s)+2),s))
	)*0.25 )

	template<typename TL>
	inline auto mapto(Int2Type<EDGE>, Field<Geometry<this_type, FACE>, TL> const &l, index_type s) const
	DECL_RET_TYPE( (
			get(l, s)+
			get(l, Shift(DES(GetM(s)+1),s))+
			get(l, Shift(DES(GetM(s)+2),s))+
			get(l, Shift(DES(GetM(s)+1)|DES(GetM(s)+2),s))+

			get(l, Shift(INC(GetM(s)),s))+
			get(l, Shift(INC(GetM(s))|DES(GetM(s)+1),s))+
			get(l, Shift(INC(GetM(s))|DES(GetM(s)+2),s))+
			get(l, Shift(INC(GetM(s))|DES(GetM(s)+1)|DES(GetM(s)+2),s))
	)*0.125 )

	template<typename TL>
	inline auto mapto(Int2Type<VOLUME>, Field<Geometry<this_type, FACE>, TL> const &l, index_type s) const
	DECL_RET_TYPE( (get(l, s)+ get(l, Shift(INC(GetM(s)),s)) )*0.5 )

	template<typename TL>
	inline auto mapto(Int2Type<VERTEX>, Field<Geometry<this_type, VOLUME>, TL> const &l, index_type s) const
	DECL_RET_TYPE(
	(
			get(l, s)+
			get(l, Shift(DES(0),s))+
			get(l, Shift(DES(1),s))+
			get(l, Shift(DES(0)|DES(1),s))+

			get(l, Shift(DES(2),s))+
			get(l, Shift(DES(2)|DES(0),s))+
			get(l, Shift(DES(2)|DES(1),s))+
			get(l, Shift(DES(2)|DES(0)|DES(1),s))
	)*0.125 )

	template<typename TL>
	inline auto mapto(Int2Type<EDGE>, Field<Geometry<this_type, VOLUME>, TL> const &l, index_type s) const
	DECL_RET_TYPE(
	(
			get(l,s)+
			get(l,Shift(DES(GetM(s)+1),s))+
			get(l,Shift(DES(GetM(s)+2),s))+
			get(l,Shift(DES(GetM(s)+1)|DES(GetM(s)+2),s))

	)*0.25 )

	template<typename TL>
	inline auto mapto(Int2Type<FACE>, Field<Geometry<this_type, VOLUME>, TL> const &l, index_type s) const
	DECL_RET_TYPE( (get(l,s)+ get(l,Shift(DES(GetM(s)),s)) )*0.5 )

//-----------------------------------------
// Vector Arithmetic
//-----------------------------------------
	nTuple<NUM_OF_DIMS, Real> dS_[2] =
	{	0, 0, 0, 0, 0, 0};
	Real dL(index_type s, int m) const
	{
		return dS_[0][m];
	}
	Real dR(index_type s, int m) const
	{
		return dS_[1][m];
	}
	template<typename TExpr, typename ... IDXS> inline auto OpEval(Int2Type<GRAD>,
	Field<Geometry<this_type, VERTEX>, TExpr> const & f, index_type s) const
	DECL_RET_TYPE( ( get(f,Shift(INC(GetM(s)),s))* dL(s,GetM(s))) +get(f, s)* dR(s,GetM(s)))

	template<typename TExpr, typename ...IDX> inline auto OpEval(Int2Type<DIVERGE>,
	Field<Geometry<this_type, EDGE>, TExpr> const & f, index_type s) const
	DECL_RET_TYPE((

			(get(f, s+PutM(0))* dL(s,0) + get(f,Shift( DX,s)+PutM(0))* dR(s,0)) +

			(get(f, s+PutM(1)) * dL(s,1) + get(f, Shift( DY,s)+PutM(1))* dR(s,1)) +

			(get(f, s+PutM(2)) * dL(s,2) + get(f, Shift( DZ,s)+PutM(2))* dR(s,2))
	))

	template<typename TL> inline auto OpEval(Int2Type<CURL>, Field<Geometry<this_type, EDGE>, TL> const & f,
	index_type s) const
	DECL_RET_TYPE((
			get(f, Shift(INC(GetM(s) + 1) ,CycleComp<2>(s) ) ) * dL(s,(GetM(s) + 1) % 3)+

			get(f, CycleComp<2>(s)) * dR(s,(GetM(s) + 1) % 3)-

			get(f, Shift(INC(GetM(s) + 2) ,CycleComp<1>(s) ) ) * dL(s,(GetM(s) + 2) % 3)-

			get(f, CycleComp<1>(s) ) * dR(s,(GetM(s) + 2) % 3)
	))

	template<typename TL> inline auto OpEval(Int2Type<CURL>, Field<Geometry<this_type, FACE>, TL> const & f,
	index_type s) const
	DECL_RET_TYPE((

			get(f, CycleComp<2>(s) ) * dL(s,(GetM(s) + 1) % 3) +

			get(f, Shift(DES(GetM(s) + 1) ,CycleComp<2>(s) ) ) * dR(s,(GetM(s) + 1) % 3) -

			get(f, CycleComp<1>(s) ) * dL(s,(GetM(s) + 2) % 3) -

			get(f, Shift(DES(GetM(s) + 1) ,CycleComp<1>(s) ) ) * dR(s,(GetM(s) + 2) % 3)

	))

//	DECL_RET_TYPE((
//	get(f,(m+2)%3,s)* dL(s)[(m + 1) % 3]
//
//	+ get(f,(m+2)%3,Shift(DES(m+1),s)) * dR(s)[(m + 1) % 3]
//
//	- get(f,(m+1)%3,s)* dL(s)[(m + 2) % 3]
//
//	- get(f,(m+1)%3,Shift(DES(m+2),s)) * dR(s)[(m + 2) % 3]
//
//	))
//
//	template<typename TL> inline auto OpEval(Int2Type<CURLPDX>,
//	Field<Geometry<this_type, EDGE>, TL> const & f, index_type s) const
//	DECL_RET_TYPE((
//	(get(f,(m==0?0:(m==1?2:1)),Shift(IX,s)) * dL(s,0)
//	+ get(f,(m==0?0:(m==1?2:1)),s)* dR(s,0))*(m==0?0:(m==1?-1:1))
//	))
//
//	template<typename TL> inline auto OpEval(Int2Type<CURLPDY>,
//	Field<Geometry<this_type, EDGE>, TL> const & f, index_type s) const
//	DECL_RET_TYPE((
//	(get(f,(m==1?0:(m==2?0:2)),Shift(IY,s)) * dL(s,1)
//	+ get(f,(m==1?0:(m==2?0:2)),s)* dR(s,1))*(m==1?0:(m==2?-1:1))
//	))
//
//	template<typename TL> inline auto OpEval(Int2Type<CURLPDZ>,
//	Field<Geometry<this_type, EDGE>, TL> const & f, index_type s) const
//	DECL_RET_TYPE((
//	(get(f,(m==2?0:(m==0?1:0)),Shift(IZ,s)) * dL(s,2)
//	+ get(f,(m==2?0:(m==0?1:0)),s)* dR(s,2))*(m==2?0:(m==0?-1:1))
//	))
//
//	template<typename TL> inline auto OpEval(Int2Type<CURLPDX>,
//	Field<Geometry<this_type, FACE>, TL> const & f, index_type s) const
//	DECL_RET_TYPE((
//	(get(f,(m==0?0:(m==1?2:1)),s) * dL(s,0)
//	+ get(f,(m==0?0:(m==1?2:1)),Shift(DX,s))* dR(s,0))*(m==0?0:(m==1?-1:1))
//	))
//	template<typename TL> inline auto OpEval(Int2Type<CURLPDY>,
//	Field<Geometry<this_type, FACE>, TL> const & f, index_type s) const
//	DECL_RET_TYPE((
//	(get(f,(m==1?0:(m==2?0:2)),s) * dL(s,1)
//	+ get(f,(m==1?0:(m==2?0:2)),Shift(DY,s))* dR(s,1))*(m==1?0:(m==2?-1:1))
//	))
//
//	template<typename TL> inline auto OpEval(Int2Type<CURLPDZ>,
//	Field<Geometry<this_type, FACE>, TL> const & f, index_type s) const
//	DECL_RET_TYPE((
//	(get(f,(m==2?0:(m==0?1:0)),s) * dL(s,2)
//	+ get(f,(m==2?0:(m==0?1:0)),Shift(DZ,s))
//	* dR(s,2))*(m==2?0:(m==0?-1:1))
//	))
//
//	template<int N, typename TL, typename ... IDXS> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,
//	Field<Geometry<this_type, N>, TL> const & f, index_type s)
//	DECL_RET_TYPE((get(f,s)*dS_[m]))

	template<int IL, int IR, typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,
	Field<Geometry<this_type, IL>, TL> const &l, Field<Geometry<this_type, IR>, TR> const &r, index_type s) const
	DECL_RET_TYPE( ( mapto(Int2Type<IL+IR>(),l,s)*
			mapto(Int2Type<IL+IR>(),r,s)))

	template<int IL, typename TL> inline auto OpEval(Int2Type<HODGESTAR>, Field<Geometry<this_type, IL>, TL> const & f,
	index_type s) const
	DECL_RET_TYPE(( mapto(Int2Type<this_type::NUM_OF_DIMS-IL >(),f,s)))

	/**
	 * non-standard Vector Field operator
	 *
	 */
	template<typename TL, typename TR> inline auto OpEval(Int2Type<DOT>, Field<Geometry<this_type, EDGE>, TL> const &l,
	Field<Geometry<this_type, EDGE>, TR> const &r, index_type s) const
	DECL_RET_TYPE( (
			mapto(Int2Type<VERTEX>(),l,0,s)* mapto(Int2Type<VERTEX>(),r,0,s)+
			mapto(Int2Type<VERTEX>(),l,1,s)* mapto(Int2Type<VERTEX>(),r,1,s)+
			mapto(Int2Type<VERTEX>(),l,2,s)* mapto(Int2Type<VERTEX>(),r,2,s)
	))

	template<typename TL, typename TR> inline auto OpEval(Int2Type<DOT>, Field<Geometry<this_type, FACE>, TL> const &l,
	Field<Geometry<this_type, FACE>, TR> const &r, index_type s) const
	DECL_RET_TYPE( (
			mapto(Int2Type<VERTEX>(),l,0,s)* mapto(Int2Type<VERTEX>(),r,0,s)+
			mapto(Int2Type<VERTEX>(),l,1,s)* mapto(Int2Type<VERTEX>(),r,1,s)+
			mapto(Int2Type<VERTEX>(),l,2,s)* mapto(Int2Type<VERTEX>(),r,2,s)
	))

	template<int IF, typename TL> inline auto OpEval(Int2Type<MAPTO0>, Field<Geometry<this_type, IF>, TL> const &l,
	index_type s) const
	DECL_RET_TYPE( (
			nTuple<3,typename Field<Geometry<this_type, IF>,TL>::value_type>( mapto(Int2Type<VERTEX>(),l,s),
					mapto(Int2Type<VERTEX>(),l,s),
					mapto(Int2Type<VERTEX>(),l,s))
	))

}
;

std::ostream & operator<<(std::ostream & os, TopologyRect const & d);
//******************************************************************************************************
// Offline define
//******************************************************************************************************

template<typename TDict> inline void TopologyRect::Load(TDict const &dict)
{

	dict["Topology"]["Dimensions"].as(&dims_);

	Update();

	LOGGER << "Load Mesh [" << GetTypeName() << "] " << DONE;
}

template<typename ...Args>
void TopologyRect::ParallelTraversal(Args const &...args) const
{
	const unsigned int num_threads = std::thread::hardware_concurrency();

	std::vector<std::thread> threads;

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		threads.emplace_back(
		        std::thread([num_threads,thread_id,this](Args const & ...args2)
		        {	this-> _Traversal(num_threads,thread_id,std::forward<Args const&>(args2)...);},
		                std::forward<Args const &>(args)...));
	}

	for (auto & t : threads)
	{
		t.join();
	}

}

template<typename ...Args>
void TopologyRect::SerialTraversal(Args const &...args) const
{
//	_Traversal(1, 0, std::forward<Args const&>( args)...);
}

}
// namespace simpla

#endif /* HEX_MESH_H_ */
