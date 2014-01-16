/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * fetl/grid/uniform_rect.h
 *
 * Created on: 2009-4-19
 * Author: salmon
 */
#ifndef UNIFORM_RECT_H_
#define UNIFORM_RECT_H_

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

#include "../fetl/field.h"
#include "../fetl/ntuple_ops.h"
#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../physics/constants.h"
#include "../physics/physical_constants.h"
#include "../utilities/log.h"
#include "../utilities/memory_pool.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/type_utilites.h"
#include "../utilities/utilities.h"
#include "media_tag.h"
namespace simpla
{

template<typename TM> class MediaTag;

/**
 *
 *  @brief UniformRectMesh -- Uniform rectangular structured grid.
 *  @ingroup mesh 
 * */

template<typename TS = Real>
struct CoRectMesh
{
	typedef CoRectMesh this_type;

	static constexpr int MAX_NUM_NEIGHBOUR_ELEMENT = 12;
	static constexpr int MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr int NUM_OF_DIMS = 3;
	static constexpr int NUM_OF_COMPONENT_TYPE = NUM_OF_DIMS + 1;

	typedef long index_type;

	typedef TS scalar_type;

	typedef nTuple<3, Real> coordinates_type;

	Real dt_ = 0.0; //!< time step

	// Topology
	unsigned int DEFAULT_GHOST_WIDTH = 2;

	nTuple<NUM_OF_DIMS, index_type> shift_ =
	{ 0, 0, 0 };

	nTuple<NUM_OF_DIMS, index_type> dims_ =
	{ 10, 10, 10 }; //!< number of cells

	nTuple<NUM_OF_DIMS, index_type> ghost_width_ =
	{ DEFAULT_GHOST_WIDTH, DEFAULT_GHOST_WIDTH, DEFAULT_GHOST_WIDTH };

	nTuple<NUM_OF_DIMS, index_type> strides_ =
	{ 0, 0, 0 };

	index_type num_cells_ = 0;

	index_type num_grid_points_ = 0;

	// Geometry
	coordinates_type xmin_ =
	{ 0, 0, 0 };
	coordinates_type xmax_ =
	{ 10, 10, 10 };

	nTuple<NUM_OF_DIMS, scalar_type> dS_[2] =
	{ 0, 0, 0, 0, 0, 0 };
	nTuple<NUM_OF_DIMS, scalar_type> k_ =
	{ 0, 0, 0 };

	coordinates_type dx_ =
	{ 0, 0, 0 };
	coordinates_type inv_dx_ =
	{ 0, 0, 0 };

	Real cell_volume_ = 1.0;
	Real d_cell_volume_ = 1.0;

	const int num_comps_per_cell_[NUM_OF_COMPONENT_TYPE] =
	{ 1, 3, 3, 1 };

	coordinates_type coordinates_shift_[NUM_OF_COMPONENT_TYPE][NUM_OF_DIMS];

	enum
	{
		C_ORDER,  //FAST_LAST

		FORTRAN_ORDER //FAST_FIRST

	};

	int array_order_;

	CoRectMesh(int array_order = FORTRAN_ORDER);

	~CoRectMesh();

	this_type & operator=(const this_type&) = delete;

	//***************************************************************************************************
	//* Media Tags
	//***************************************************************************************************

	typedef MediaTag<this_type> tag_container;

	typedef MediaTag<this_type> media_tag_type;

	typedef typename MediaTag<this_type>::tag_type tag_type;

private:
	std::shared_ptr<tag_container> tags_;
public:

	tag_container & tags()
	{
		if (tags_ == nullptr)
			tags_ = std::shared_ptr<tag_container>(new tag_container(*this));

		return *tags_;
	}
	tag_container const& tags() const
	{
		if (tags_ == nullptr)
			ERROR << "Media Tag is not initialized!!";
		return *tags_;
	}

	//***************************************************************************************************
	//* Constants
	//***************************************************************************************************
private:
	std::shared_ptr<PhysicalConstants> constants_;	//!< Unit System and phyical constants
public:
	PhysicalConstants & constants()
	{
		if (constants_ == nullptr)
			constants_ = std::shared_ptr<PhysicalConstants>(new PhysicalConstants());

		return *constants_;
	}
	PhysicalConstants const& constants() const
	{
		if (constants_ == nullptr)
			ERROR << "Constants are not defined!!";
		return *constants_;
	}

	//***************************************************************************************************
	//* Configure
	//***************************************************************************************************

	template<typename ISTREAM> void Deserialize(ISTREAM const &cfg);

	template<typename OSTREAM> OSTREAM& Serialize(OSTREAM &vm) const;

	void Update();

	bool CheckCourant(Real dt)
	{
		DEFINE_PHYSICAL_CONST((*constants_));

		Real res = 0.0;

		for (int i = 0; i < 3; ++i)
			res += inv_dx_[i] * inv_dx_[i];
		res = std::sqrt(res) * speed_of_light * dt;

		if (res > 1.0)
			WARNING << "Not statisfy Courant Conditions!";

		return res < 1.0;
	}

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}

	static inline std::string GetTypeName()
	{
		return "CoRectMesh";
	}

	inline std::string GetTopologyTypeAsString() const
	{
		return ToString(GetRealNumDimension()) + "DCoRectMesh";
	}

	// Properties of UniformRectMesh --------------------------------------
	inline void SetGhostWidth(int i, index_type v)
	{
		ghost_width_[i % NUM_OF_DIMS] = v;
	}

	inline nTuple<NUM_OF_DIMS, index_type> const&GetGhostWidth() const
	{
		return ghost_width_;
	}

//	inline void SetExtent(coordinates_type const & pmin, coordinates_type const & pmax)
//	{
//		xmin_ = pmin;
//		xmax_ = pmax;
//	}
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

//	inline void SetDimension(nTuple<NUM_OF_DIMS, index_type> const & pdims)
//	{
//		dims_ = pdims;
//	}

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

	inline nTuple<NUM_OF_DIMS, index_type> const & GetDimension() const
	{
		return dims_;
	}

	inline int GetRealNumDimension() const
	{
		int n = 0;
		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			if (dims_[i] > 1)
				++n;
		}
		return n;
	}
	inline std::vector<index_type> GetShape(int IFORM) const
	{
		std::vector<index_type> res;
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
	inline nTuple<NUM_OF_DIMS, index_type> const & GetStrides() const
	{
		return strides_;
	}

	// General Property -----------------------------------------------

	inline Real GetDt() const
	{
		return dt_;
	}

	inline void SetDt(Real dt = 0.0)
	{
		dt_ = dt;
		Update();
	}

	inline index_type GetNumOfElements(int iform) const
	{

		return (num_grid_points_ * num_comps_per_cell_[iform]);
	}

	inline index_type GetNumOfVertices(...) const
	{

		return (num_grid_points_);
	}

	inline Real GetCellVolume(...) const
	{
		return cell_volume_;
	}

	inline Real GetDCellVolume(...) const
	{
		return d_cell_volume_;
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

	inline coordinates_type GetCoordinates(int IFORM, int m, index_type i, index_type j, index_type k) const
	{

		coordinates_type res = xmin_;
		res[0] += i * dx_[0] + coordinates_shift_[IFORM][m][0];
		res[1] += j * dx_[1] + coordinates_shift_[IFORM][m][1];
		res[2] += k * dx_[2] + coordinates_shift_[IFORM][m][2];
		return std::move(res);
	}

	inline coordinates_type GetCoordinates(int IFORM, int m, index_type s) const
	{

		coordinates_type res = xmin_;

		index_type idx[3];

		UnpackIndex(idx,s);

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			res[i] =idx[i]*dx_[i]+ coordinates_shift_[IFORM][m][i];
		}
		return std::move(res);
	}

	inline coordinates_type GetCoordinates(int iform, index_type s) const
	{
		return std::move(GetCoordinates(iform, int(s % num_comps_per_cell_[iform]), s / num_comps_per_cell_[iform]));

	}

	inline coordinates_type GetGlobalCoordinates(index_type s, coordinates_type const &r) const
	{
		return GetCoordinates(0, s) + r * dx_;
	}

	template<int IFORM>
	inline index_type GetSubComponent(index_type s) const
	{
		return s % num_comps_per_cell_[IFORM];
	}

	template<int IFORM>
	inline index_type GetCellIndex(index_type s) const
	{
		return (s-s % num_comps_per_cell_[IFORM])/num_comps_per_cell_[IFORM];
	}

	template<typename ... IDXS>
	inline index_type GetComponentIndex(int IFORM, int m, IDXS ... s) const
	{
		return GetIndex(s...) * num_comps_per_cell_[IFORM] + m;
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

		for (int i = 0; i < 3; ++i)
		{
			if (dx_[i] > 0)
			{
				s += static_cast<index_type>(std::floor((x[i] - xmin_[i]) / dx_[i])) * strides_[i];
			}
		}
		return s;
	}
private:

	/**
	 *
	 * @param
	 * @param
	 * @param v
	 * @param m
	 * @param s
	 * @return
	 */
	template<int I, typename ... Args>
	inline int _GetNeighbourCell(Int2Type<I>, Int2Type<I>, index_type *v, int m, Args ... s) const
	{
		if (v != nullptr)
		v[0] = GetIndex(s...);
		return 1;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<EDGE>, Int2Type<VERTEX>, index_type *v, int m, Args ... s) const
	{
		v[0] = GetIndex(s...);
		v[1] = Shift(INC(m), s...);
		return 2;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<FACE>, Int2Type<VERTEX>, index_type *v, int m, Args ... s) const
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
			v[0] = GetIndex(s...);
			v[1] = Shift(INC(m + 1), s...);
			v[2] = Shift(INC(m + 1) | INC(m + 2), s...);
			v[3] = Shift(INC(m + 2), s...);
		}
		return 4;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<VOLUME>, Int2Type<VERTEX>, index_type *v, int m, Args ... s) const
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
			v[0] = GetIndex(s...);
			v[1] = Shift(INC(0), s...);
			v[2] = Shift(INC(1) | INC(1), s...);
			v[3] = Shift(INC(1), s...);

			v[4] = Shift(INC(2), s...);
			v[5] = Shift(INC(2) | INC(0), s...);
			v[6] = Shift(INC(2) | INC(1) | INC(1), s...);
			v[7] = Shift(INC(2) | INC(1), s...);
		}
		return 8;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<VERTEX>, Int2Type<EDGE>, index_type *v, int m, Args ... s) const
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
			v[0] = GetComponentIndex(1, 0, s...);
			v[1] = GetComponentIndex(1, 1, s...);
			v[2] = GetComponentIndex(1, 2, s...);
			v[3] = GetComponentIndex(1, 0, Shift(DES(0), s...));
			v[4] = GetComponentIndex(1, 1, Shift(DES(1), s...));
			v[5] = GetComponentIndex(1, 2, Shift(DES(2), s...));
		}
		return 6;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<FACE>, Int2Type<EDGE>, index_type *v, int m, Args ... s) const
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
			v[0] = GetComponentIndex(1, (m + 1) % 3, s...);
			v[1] = GetComponentIndex(1, (m + 2) % 3, s...);
			v[2] = GetComponentIndex(1, (m + 2) % 3, Shift(INC(m + 1), s...));
			v[2] = GetComponentIndex(1, (m + 1) % 3, Shift(INC(m + 2), s...));
		}
		return 4;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<VOLUME>, Int2Type<EDGE>, index_type *v, int m, Args ... s) const
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
			v[0] = GetComponentIndex(1, 0, s...);
			v[1] = GetComponentIndex(1, 1, Shift(INC(0), s...));
			v[2] = GetComponentIndex(1, 0, Shift(INC(1), s...));
			v[3] = GetComponentIndex(1, 1, s...);

			v[4] = GetComponentIndex(1, 2, s...);
			v[5] = GetComponentIndex(1, 2, Shift(INC(0), s...));
			v[6] = GetComponentIndex(1, 2, Shift(INC(1), s...));
			v[7] = GetComponentIndex(1, 2, s...);

			v[8] = GetComponentIndex(1, 0, Shift(INC(2), s...));
			v[9] = GetComponentIndex(1, 1, Shift(INC(2) | INC(0), s...));
			v[10] = GetComponentIndex(1, 0, Shift(INC(2) | INC(1), s...));
			v[11] = GetComponentIndex(1, 1, Shift(INC(2), s...));

		}
		return 12;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<VERTEX>, Int2Type<FACE>, index_type *v, int m, Args ... s) const
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
			v[0] = GetComponentIndex(2, 0, s...);
			v[1] = GetComponentIndex(2, 0, Shift(DES(2), s...));
			v[2] = GetComponentIndex(2, 0, Shift(DES(2) | DES(1), s...));
			v[3] = GetComponentIndex(2, 0, Shift(DES(1), s...));

			v[4] = GetComponentIndex(2, 1, s...);
			v[5] = GetComponentIndex(2, 1, Shift(DES(2), s...));
			v[6] = GetComponentIndex(2, 1, Shift(DES(0) | DES(2), s...));
			v[7] = GetComponentIndex(2, 1, Shift(DES(0), s...));

			v[8] = GetComponentIndex(2, 2, s...);
			v[9] = GetComponentIndex(2, 2, Shift(DES(1), s...));
			v[10] = GetComponentIndex(2, 2, Shift(DES(1) | DES(0), s...));
			v[11] = GetComponentIndex(2, 2, Shift(DES(0), s...));

		}
		return 12;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<EDGE>, Int2Type<FACE>, index_type *v, int m, Args ... s) const
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
			v[0] = GetComponentIndex(2, (m + 1) % 3, s...);
			v[1] = GetComponentIndex(2, (m + 2) % 3, s...);
			v[2] = GetComponentIndex(2, (m + 1) % 3, Shift(DES(m + 2), s...));
			v[2] = GetComponentIndex(2, (m + 2) % 3, Shift(DES(m + 1), s...));
		}
		return 4;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<VOLUME>, Int2Type<FACE>, index_type *v, int m, Args ... s) const
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
			v[0] = GetComponentIndex(2, 0, s...);
			v[1] = GetComponentIndex(2, 1, Shift(INC(1), s...));
			v[2] = GetComponentIndex(2, 0, Shift(INC(0), s...));
			v[3] = GetComponentIndex(2, 1, s...);

			v[4] = GetComponentIndex(2, 2, s...);
			v[5] = GetComponentIndex(2, 2, Shift(INC(0), s...));

		}
		return 6;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<VERTEX>, Int2Type<VOLUME>, index_type *v, int m, Args ... s) const
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
			v[0] = GetIndex(s...);
			v[1] = Shift(DES(0), s...);
			v[2] = Shift(DES(0) | DES(1), s...);
			v[3] = Shift(DES(1), s...);

			v[4] = Shift(DES(2), s...);
			v[5] = Shift(DES(2)|DES(0), s...);
			v[6] = Shift(DES(2)|DES(0)|DES(1), s...);
			v[7] = Shift(DES(2)|DES(1), s...);

		}
		return 8;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<EDGE>, Int2Type<VOLUME>, index_type *v, int m, Args ... s) const
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
			v[0] = GetIndex( s...);
			v[1] = Shift(DES(m + 1), s...);
			v[2] = Shift(DES(m + 1)|DES(m + 2), s...);
			v[3] = Shift(DES(m + 2), s...);
		}
		return 4;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<FACE>, Int2Type<VOLUME>, index_type *v, int m, Args ... s) const
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
			v[0] = GetIndex( s...);
			v[1] = Shift(DES(m), s...);

		}
		return 2;
	}

public:

	template<int IN, int OUT>
	inline index_type GetNeighbourCell(Int2Type<IN>, Int2Type<OUT>, index_type *v, index_type s) const
	{
		return _GetNeighbourCell(Int2Type<IN>(), Int2Type<OUT>(), v,
		s%num_comps_per_cell_[IN],
		(s-s%num_comps_per_cell_[IN])/num_comps_per_cell_[IN]);

	}

	template<int IN, int OUT,typename ... Args>
	inline index_type GetNeighbourCell(Int2Type<IN>, Int2Type<OUT>, index_type *v, int m,Args const &... s) const
	{
		return _GetNeighbourCell(Int2Type<IN>(), Int2Type<OUT>(), v, m, s...);
	}

	//****************************************************************************************************
	//* Index operation
	//*
	//***************************************************************************************************

	void UnpackIndex(index_type *idx,index_type s)const
	{
		UnpackIndex(idx,idx+1,idx+2, s);
	}
	void UnpackIndex(index_type *i,index_type *j,index_type *k,index_type s)const
	{
		*i =(strides_[0]==0)?0:(s/strides_[0]);
		s-=(*i)*strides_[0];
		*j =(strides_[1]==0)?0:(s/strides_[1]);
		s-=(*j)*strides_[1];
		*k =s;
	}
	void UnpackIndex(index_type *i,index_type *j,index_type *k,index_type i1,index_type j1,index_type k1 )const
	{
		*i=i1;
		*j=j1;
		*k=k1;
	}

	/**
	 * 	Thanks my wife Dr. CHEN Xiang Lan, for her advice on  these bitwise operation
	 */

	typedef signed long shift_type;

	static constexpr int DIGITS_LONG=std::numeric_limits<unsigned long>::digits; //!< signed long is 63bit, unsigned long is 64 bit, add a sign bit

	static constexpr int DIGITS_SHORT=std::numeric_limits<unsigned short>::digits;

#define _shift_bit(m) \
	 static_cast<shift_type>((static_cast<unsigned long>((-1L) << (DIGITS_LONG - DIGITS_SHORT)) >> (DIGITS_LONG - DIGITS_SHORT*(m+1) )))
	enum
	{
		X = 1L, // 0000 0000 0001
		NX= _shift_bit(0) ,// 0000 0000 1111
		Y = 1L<<DIGITS_SHORT,// 0000 0001 0000
		NY=_shift_bit(1),// 0000 1111 0000
		Z = 1L<<(DIGITS_SHORT*2),// 0001 0000 0000
		NZ=_shift_bit(2)// 1111 0000 0000
	};

	inline shift_type INC(int m) const
	{
		return 1L << ((m % 3) * DIGITS_SHORT);
	}
	inline shift_type DES(int m) const
	{
		return _shift_bit((m%3));
	}

#undef _shift_bit
	inline index_type Shift(shift_type d, index_type i, index_type j, index_type k) const
	{

//		auto ix = (d << (DIGITS_OF_COMPACT_SHIFT - DIGITS_OF_SHIFT*1)) >> (DIGITS_OF_COMPACT_SHIFT - DIGITS_OF_SHIFT);
//		auto jx = (d << (DIGITS_OF_COMPACT_SHIFT - DIGITS_OF_SHIFT*2)) >> (DIGITS_OF_COMPACT_SHIFT - DIGITS_OF_SHIFT);
//		auto kx = (d << (DIGITS_OF_COMPACT_SHIFT - DIGITS_OF_SHIFT*3)) >> (DIGITS_OF_COMPACT_SHIFT - DIGITS_OF_SHIFT);
//
//		CHECK(ix)<<" "<<jx<<" "<<kx<<" "<<std::hex<<d;

		i += (d << (DIGITS_LONG - DIGITS_SHORT*1)) >> (DIGITS_LONG - DIGITS_SHORT);
		j += (d << (DIGITS_LONG - DIGITS_SHORT*2)) >> (DIGITS_LONG - DIGITS_SHORT);
		k += (d << (DIGITS_LONG - DIGITS_SHORT*3)) >> (DIGITS_LONG - DIGITS_SHORT);

		return GetIndex(i,j,k);
	}

	template<typename ... IDXS>
	inline index_type Shift(shift_type d, IDXS ... s) const
	{
		index_type i,j,k;
		UnpackIndex(&i,&j,&k,s...);
		return Shift(d,i,j,k);
	}

	inline index_type GetIndex(index_type i, index_type j, index_type k) const
	{

		auto res= (

		(((i % dims_[0])+dims_[0]) % dims_[0]) * strides_[0]+

		(((j % dims_[1])+dims_[1]) % dims_[1]) * strides_[1]+

		(((k % dims_[2])+dims_[2]) % dims_[2]) * strides_[2]

		);
		return res;
	}

	inline index_type GetIndex(index_type* i) const
	{
		return GetIndex(i[0],i[1],i[2]);
	}

	inline index_type GetIndex(index_type s) const
	{
		return s;
	}

	//***************************************************************************************************
	//  Traversal
	//
	//***************************************************************************************************

	template<typename ... Args>
	void Traversal(Args const &...args) const
	{
		ParallelTraversal(std::forward<Args const &>(args)... );
	}

	template<typename ...Args> void ParallelTraversal(Args const &...args)const;

	template<typename ...Args> void SerialTraversal(Args const &...args)const;

	void _Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
	std::function<void(int, index_type, index_type, index_type)> const &funs) const;

	void _Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
	std::function<void(index_type)> const &fun) const
	{
		_Traversal(num_threads,thread_id,
		IFORM, [&](int m,index_type i,index_type j,index_type k)
		{
			fun(GetComponentIndex(IFORM,m,i,j,k));
		});

	}
	void _Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
	std::function<void(index_type, coordinates_type)> const &fun ) const
	{
		int num = num_comps_per_cell_[IFORM];

		_Traversal(num_threads,thread_id,
		IFORM, [&](int m,index_type i,index_type j,index_type k)
		{
			fun(GetComponentIndex(IFORM,m,i,j,k),this->GetCoordinates(IFORM,m,i,j,k));
		});

	}

	template<typename Fun, typename TF, typename ... Args> inline
	void SerialForEach(Fun const &fun, TF const & l, Args const& ... args) const
	{
		SerialTraversal(FieldTraits<TF>::IForm, [&](int m,index_type i,index_type j,index_type k)
		{	fun(get(l,m,i,j,k),get(args,m,i,j,k)...);});
	}

	template<typename Fun, typename TF, typename ... Args> inline
	void SerialForEach(Fun const &fun, TF *l, Args const& ... args) const
	{
		if (l==nullptr)
		ERROR << "Access value to an uninitilized container!";

		SerialTraversal(FieldTraits<TF>::IForm, [&](int m,index_type i,index_type j,index_type k)
		{	fun(get(l,m,i,j,k),get(args,m,i,j,k)...);});
	}

	template<typename Fun, typename TF, typename ... Args> inline
	void ParallelForEach(Fun const &fun, TF const & l, Args const& ... args) const
	{
		ParallelTraversal(FieldTraits<TF>::IForm, [&](int m,index_type i,index_type j,index_type k)
		{	fun(get(l,m,i,j,k),get(args,m,i,j,k)...);});
	}

	template<typename Fun, typename TF, typename ... Args> inline
	void ParallelForEach(Fun const &fun, TF *l, Args const& ... args) const
	{
		if (l==nullptr)
		ERROR << "Access value to an uninitilized container!";

		ParallelTraversal(FieldTraits<TF>::IForm, [&](int m,index_type i,index_type j,index_type k)
		{	fun(get(l,m,i,j,k),get(args,m,i,j,k)...);});
	}

	template<typename Fun, typename TF, typename ... Args> inline
	void ForEach(Fun const &fun, TF const & l, Args const& ... args) const
	{
		ParallelForEach(fun,l,std::forward<Args const &>(args)...);
	}
	template<typename Fun, typename TF, typename ... Args> inline
	void ForEach(Fun const &fun, TF *l, Args const& ... args) const
	{
		ParallelForEach(fun,l,std::forward<Args const &>(args)...);
	}

	//***************************************************************************************************
	//* Container/Field operation
	//* Field vs. Mesh
	//***************************************************************************************************

	template<typename TV>
	void SetFieldValue(Field<Geometry<this_type,1> ,TV> * f,nTuple<3,TV> const &v,index_type s)const
	{
		(*f)[s*3]=v[0];
		(*f)[s*3+1]=v[1];
		(*f)[s*3+2]=v[2];
	}

	template<typename TV>
	void SetFieldValue(Field<Geometry<this_type,2> ,TV> * f,nTuple<3,TV> const &v,index_type s)const
	{
		(*f)[s*3]=v[0];
		(*f)[s*3+1]=v[1];
		(*f)[s*3+2]=v[2];
	}

	template<typename TV>
	void SetFieldValue(Field<Geometry<this_type,0> ,TV> * f,TV const &v,index_type s)const
	{
		(*f)[s]=v;
	}

	template<typename TV>
	void SetFieldValue(Field<Geometry<this_type,3> ,TV> * f,TV const &v,index_type s)const
	{
		(*f)[s]=v;
	}

	template<typename TL, typename TR> void AssignContainer(int IFORM, TL * lhs, TR const &rhs) const
	{
		ParallelTraversal(IFORM, [&](int m, index_type x, index_type y, index_type z)
		{	get(lhs,m,x,y,z)=get(rhs,m,x,y,z);});

	}

	template<typename TL,int IL>
	typename std::enable_if<IL==1|IL==2,void>::type AssignContainer(Field<Geometry<this_type,IL> ,TL> * lhs,
	typename Field<Geometry<this_type,IL> ,TL>::field_value_type const &rhs) const
	{
		ParallelTraversal(0, [&](int m, index_type x, index_type y, index_type z)
		{
			get(lhs,0,x,y,z)=rhs[0];
			get(lhs,1,x,y,z)=rhs[1];
			get(lhs,2,x,y,z)=rhs[2];

		});
	}

	template<typename TL,int IL>
	typename std::enable_if<IL==0|IL==3,void>::type AssignContainer(Field<Geometry<this_type,IL> ,TL> * lhs,
	typename Field<Geometry<this_type,IL> ,TL>::field_value_type const &rhs) const
	{
		ParallelTraversal(0, [&](int m, index_type x, index_type y, index_type z)
		{	get(lhs,0,x,y,z)=rhs;});
	}

	template<typename T, typename ... TI>
	inline typename std::enable_if<!is_field<T>::value, T>::type get(T const &l, TI ...) const
	{
		return std::move(l);
	}

	template<int IFORM, typename TL> inline typename Field<Geometry<this_type, IFORM>, TL>::value_type & get(
	Field<Geometry<this_type, IFORM>, TL> *l, index_type s) const
	{
		return l->get(s % num_comps_per_cell_[IFORM], s / num_comps_per_cell_[IFORM]);
	}

	template<int IFORM, typename TL, typename ...TI> inline typename Field<Geometry<this_type, IFORM>, TL>::value_type & get(
	Field<Geometry<this_type, IFORM>, TL> *l, TI ... s) const
	{
		return l->get(s...);
	}

	template<int IFORM, typename TL, typename ... TI>
	typename Field<Geometry<this_type, IFORM>, TL>::value_type const &get(
	Field<Geometry<this_type, IFORM>, TL> const & l, TI ...s) const
	{
		return (l.get(s...));
	}

	template<int IFORM, int TOP, typename TL, typename ... TI>
	typename Field<Geometry<this_type, IFORM>, UniOp<TOP, TL> >::value_type get(
	Field<Geometry<this_type, IFORM>, UniOp<TOP, TL> > const & l, TI ...s) const
	{
		return (l.get(s...));
	}

	template<int IFORM, int TOP, typename TL, typename TR, typename ... TI>
	typename Field<Geometry<this_type, IFORM>, BiOp<TOP, TL, TR> >::value_type get(
	Field<Geometry<this_type, IFORM>, BiOp<TOP, TL, TR> > const & l, TI ...s) const
	{
		return (l.get(s...));
	}

	template<typename TV>
	TV & get_value(Container<TV> & d,index_type s)const
	{
		return * (d.get()+s);
	}
	template<typename TV>
	TV const & get_value(Container<TV> const& d,index_type s)const
	{
		return * (d.get()+s);
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
	inline index_type SearchCell(coordinates_type const &x, Real * r=nullptr) const
	{

		index_type s[3];

		for(int n=0;n<3;++n)
		{
			Real L=(xmax_[n] - xmin_[n]);

			double i=0,e=0;

			if(L>0) e= std::modf( std::fmod( std::fmod( x[n] - xmin_[n],L )+L , L) *inv_dx_[n],&i);

			if(r!=nullptr)
			{
				r[n]=e;
			}

			s[n]=static_cast<index_type>(i);

		}

		return GetIndex(s);
	}

	/**
	 *
	 * Locate the cell containing a specified point.
	 * and apply cycle boundary condition on x
	 *
	 * @param x
	 * @param r
	 * @return index of cell
	 */
	inline index_type SearchCell(Real *x, Real *r =nullptr) const
	{

		index_type s[3];

		for(int n=0;n<3;++n)
		{
			Real L=(xmax_[n] - xmin_[n]);

			x[n]=

			(xmax_[n]<=xmin_[n])

			?0

			:std::fmod( std::fmod( x[n] - xmin_[n],L )+L , L);

			double i,e;

			e= std::modf(x[n]*inv_dx_[n],&i);

			if(r!=nullptr) r[n]=e;

			s[n]=static_cast<index_type>(i);

			x[n]+=xmin_[n];

		}

		return GetIndex(s);
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

	inline index_type SearchCell(index_type const &hint_idx, Real * x, Real *pcoords = nullptr) const
	{
		return SearchCell(x, pcoords);
	}

	inline index_type SearchCellFix(index_type s, Real *r)const
	{
		shift_type d=0;
		for(int i=0;i<3;++i)
		{
			if(dims_[i]<=1)
			{
				r[i]=0;
			}
			else if(r[i]<0)
			{
				double n;
				r[i]=std::modf(r[i],&n)+1;
				d|=DES(i)*static_cast<signed int>(n-1);
			}
			else if(r[i]>1.0)
			{
				double n;
				r[i]=std::modf(r[i],&n);
				d|=INC(i)*static_cast<signed int>(n);
			}

		}

		return Shift(d,s);

	}

	/**
	 *
	 * @param s
	 * @param x
	 * @return number vertex
	 * 	     2 for uniform rectangle
	 * 	     4 for Tetrahedron
	 */
	inline int GetCellShape(index_type s, coordinates_type * x=nullptr) const
	{
		if(x!=nullptr)
		{
			x[0]=GetCoordinates(0,s);
			x[1] =x[0]+dx_;
		}
		return 2;
	}

	index_type Refelect(index_type hint_s,Real dt,coordinates_type * x, nTuple<3,Real> * v)const
	{
		coordinates_type r;
		r=*x;
		index_type s = SearchCell(hint_s,&(r)[0]);

		shift_type d=0;

		for(int i=0;i<3;++i)
		{
			auto a=r[i]-dt*(*v)[i]*inv_dx_[i];
			if(a <0)
			{
				d|=DES(i);
			}
			else if(a >1 )
			{
				d|=INC(i);
			}
			else
			{
				continue;
			}
			v[i] *=-1;
			r[i] =1.0-(*x)[i];
		}

		if(d!=0)
		{
			*x=GetGlobalCoordinates(s,r);
			s= Shift(d,s);
		}
		return s;

	}
	//***************************************************************************************************
	// Particle <=> Mesh/Cache
	// Begin

	//  Scatter/Gather to Cache
	template<int I> inline index_type
	GetAffectedPoints(Int2Type<I>, index_type const & s=0, index_type * points=nullptr, int affect_region = 2) const
	{
		index_type i,j,k;

		UnpackIndex(&i,&j,&k,s);

		index_type num=num_comps_per_cell_[I];

		if(points!=nullptr)
		{
			int t=0;

			index_type i_b= (dims_[0]>1)?i-affect_region+1:0;
			index_type i_e= (dims_[0]>1)?i+affect_region+1:1;
			index_type j_b= (dims_[1]>1)?j-affect_region+1:0;
			index_type j_e= (dims_[1]>1)?j+affect_region+1:1;
			index_type k_b= (dims_[2]>1)?k-affect_region+1:0;
			index_type k_e= (dims_[2]>1)?k+affect_region+1:1;

			for(index_type l=i_b;l<i_e;++l)
			for(index_type m=j_b;m<j_e;++m)
			for(index_type n=k_b;n<k_e;++n)
			{
				points[t] = GetIndex(l,m,n)*num;

				for(int s=1;s<num;++s)
				{
					points[t+s]=points[t]+s;
				}
				t+=num;
			}
		}

		index_type w=1;

		for(int i=0;i<3;++i)
		{
			if(dims_[i]>1)
			{
				w*=(affect_region*2);
			}
		}
		return w*num;
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

	template<typename TV,typename TW>
	inline void ScatterToCache(Real const x[3],TW const &v , TV *cache,int w,int num_of_comp=1,int comp_num=0)const
	{
		Real r[3]=
		{	x[0], x[1], x[2]};
		index_type sx[3];
		index_type o=GetCacheCoordinates(w,sx,r);

		DEF_INTERPOLATION_SCHEME(,+=v)
	}

	template<typename TV,typename TW>
	inline void GatherFromCache(Real const x[3],TV const*cache,TW *res , int w,int num_of_comp=1,int comp_num=0)const
	{
		Real r[3]=
		{	x[0], x[1], x[2]};

		index_type sx[3];
		index_type o=GetCacheCoordinates(w,sx,r);

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

	template<typename TV,typename TW>
	inline void Gather(Int2Type<VERTEX>, Real const *rr, TV const* cache, TW* v, int w = 2) const
	{
		GatherFromCache(rr,cache,v,w );
	}

	template<typename TV,typename TW>
	inline void Scatter(Int2Type<VOLUME>,Real const *rr, TW const & v,TV* cache, int w = 2) const
	{
		Real r[3]=
		{	rr[0]-0.5, rr[1]-0.5, rr[2]-0.5};

		ScatterToCache(r,std::forward<TW const &>(v),cache,w );
	}

	template<typename TV,typename TW>
	inline void Gather(Int2Type<VOLUME>, Real const *rr, TV const* cache, TW* v, int w = 2) const
	{
		Real r[3]=
		{	rr[0]-0.5, rr[1]-0.5, rr[2]-0.5};

		GatherFromCache(r,cache,v,w );
	}

	template<typename TV,typename TW>
	inline void
	Scatter(Int2Type<EDGE>,Real const *rr, nTuple<3,TW> const & v,TV* cache, int w = 2) const
	{

		for(int m=0;m<3;++m)
		{
			Real r[3]=
			{	rr[0], rr[1], rr[2]};
			r[m]-=0.5;
			ScatterToCache(r,v[m],cache,w,3,m );

		}
	}

	template<typename TV,typename TW>
	inline void Gather(Int2Type<EDGE>, Real const *rr, TV const* cache, nTuple<3,TW>* v, int w = 2) const
	{
		(*v) = 0;
		for(int m=0;m<3;++m)
		{
			Real r[3]=
			{	rr[0], rr[1], rr[2]};
			r[m]-=0.5;
			GatherFromCache(r,cache,&(*v)[m],w,3,m );
		}
	}

	template<typename TV,typename TW>
	inline void Scatter(Int2Type<FACE>,Real const *rr, nTuple<3,TW> const & v,TV* cache, int w = 2) const
	{

		for(int m=0;m<3;++m)
		{
			Real r[3]=
			{	rr[0], rr[1], rr[2]};
			r[(m+1)%2]-=0.5;
			r[(m+2)%2]-=0.5;
			ScatterToCache(r,v[m],cache,w,3,m );

		}
	}

	template<typename TV,typename TW>
	inline void Gather(Int2Type<FACE>, Real const *rr, TV const* cache, nTuple<3,TW>* v, int w = 2) const
	{
		(*v) = 0;
		for(int m=0;m<3;++m)
		{
			Real r[3]=
			{	rr[0], rr[1], rr[2]};
			r[(m+1)%2]-=0.5;
			r[(m+2)%2]-=0.5;
			GatherFromCache(r,cache,&(*v)[m],w,3,m );
		}
	}

	template<int IFORM,typename TV,typename TR> void
	GetMeanValue(TV const * cache,TR * v,int affect_region)const
	{
		index_type w=2*affect_region;

		index_type sx[3];

		sx[0]= (dims_[0]<=1)?0:((dims_[1]<=1)?1:w*2) *((dims_[2]<=1)?1:w*2) ,

		sx[1]= (dims_[1]<=1)?0:((dims_[2]<=1)?1:w*2);

		sx[2]= (dims_[2]<=1)?0:1;

		int count=0;

		for(index_type i=0,ie=((dims_[0]>1)?w:1);i<ie;++i)
		for(index_type j=0,je=((dims_[1]>1)?w:1);j<je;++j)
		for(index_type k=0,ke=((dims_[2]>1)?w:1);k<ke;++k)
		{
			index_type s= i*sx[0]+j*sx[1]+k*sx[2];
			++count;

			for(index_type m=0;m<num_comps_per_cell_[IFORM];++m)
			{
				v[m]+=cache[s*num_comps_per_cell_[IFORM]+m];
			}
		}

		for(index_type m=0;m<num_comps_per_cell_[IFORM];++m)
		{
			v[m]/=static_cast<Real>(count);
		}

	}

	//  Scatter/Gather to Mesh

#define DEF_INTERPOLATION_SCHEME(_LEFT_,_RIGHT_)                                                       \
	_LEFT_ cache[(o)*num_of_comp+m] _RIGHT_ * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);       \
	_LEFT_ cache[Shift(X,o)*num_of_comp+m] _RIGHT_ * r[0] * (1.0 - r[1]) * (1.0 - r[2]);         \
	_LEFT_ cache[Shift(Y,o)*num_of_comp+m] _RIGHT_ * (1.0 - r[0]) * r[1]* (1.0 - r[2]);          \
	_LEFT_ cache[Shift(X|Y,o)*num_of_comp+m] _RIGHT_ * r[0] * r[1]* (1.0 - r[2]);            \
	_LEFT_ cache[Shift(Z,o)*num_of_comp+m] _RIGHT_ * (1.0 - r[0]) * (1.0 - r[1]) * r[2];         \
	_LEFT_ cache[Shift(X|Z,o)*num_of_comp+m] _RIGHT_ * r[0] * (1.0 - r[1]) * r[2];           \
	_LEFT_ cache[Shift(Y|Z,o)*num_of_comp+m] _RIGHT_ * (1.0 - r[0]) * r[1]* r[2];            \
	_LEFT_ cache[Shift(X|Y|Z,o)*num_of_comp+m] _RIGHT_ * r[0] * r[1]* r[2];

	/**
	 *
	 * @param
	 * @param o
	 * @param r is cell local coordinate not global coordiante
	 * @param m
	 * @param v
	 * @param cache
	 */
	template<typename TV,typename TW>
	inline void Scatter(Int2Type<VERTEX>,index_type o,Real const r[3], TW const &v , TV *cache)const
	{
		int num_of_comp=num_comps_per_cell_[VERTEX];
		int m=0;

		DEF_INTERPOLATION_SCHEME(,+=v)
	}

	template<typename TV,typename TW>
	inline void Gather(Int2Type<VERTEX>,index_type o,Real const r[3], TV const*cache,TW *v )const
	{
		int num_of_comp=num_comps_per_cell_[VERTEX];
		int m=0;
		(*v) = 0;
		DEF_INTERPOLATION_SCHEME((*v)+=,)
	}

	template<typename TV,typename TW>
	inline void Scatter(Int2Type<EDGE>,index_type o,Real const rr[3],TW const &v , TV *cache)const
	{
		int num_of_comp=num_comps_per_cell_[FACE];
		for(int m=0;m<num_of_comp;++m)
		{
			Real r[3] =
			{	rr[0],rr[1],rr[2]};

			r[m]-=0.5;
			o=SearchCellFix(o,r);
			DEF_INTERPOLATION_SCHEME(,+=v[m])
		}
	}

	template<typename TV,typename TW>
	inline void Gather(Int2Type<EDGE>,index_type o,Real const rr[3], TV const*cache,TW *v )const
	{
		int num_of_comp=num_comps_per_cell_[FACE];

		for(int m=0;m<num_of_comp;++m)
		{
			Real r[3] =
			{	rr[0],rr[1],rr[2]};

			r[m]-=0.5;
			(*v) = 0;
			o=SearchCellFix(o,r);
			DEF_INTERPOLATION_SCHEME((*v)[m]+=,)
		}
	}

	template<typename TV,typename TW>
	inline void Scatter(Int2Type<FACE>,index_type o,Real const rr[3],TW const &v , TV *cache)const
	{

		int num_of_comp=num_comps_per_cell_[FACE];
		for(int m=0;m<num_of_comp;++m)
		{
			Real r[3] =
			{	rr[0],rr[1],rr[2]};

			r[(m+1)%3]-=0.5;
			r[(m+2)%3]-=0.5;
			o=SearchCellFix(o,r);
			DEF_INTERPOLATION_SCHEME(,+=v[m])
		}
	}

	template<typename TV,typename TW>
	inline void Gather(Int2Type<FACE>,index_type o,Real const rr[3], TV const*cache,TW *v )const
	{

		int num_of_comp=num_comps_per_cell_[FACE];

		for(int m=0;m<num_of_comp;++m)
		{
			Real r[3] =
			{	rr[0],rr[1],rr[2]};

			r[(m+1)%3]-=0.5;
			r[(m+2)%3]-=0.5;

			(*v) = 0;
			o=SearchCellFix(o,r);
			DEF_INTERPOLATION_SCHEME((*v)[m]+=,)
		}

	}

	template<typename TV,typename TW>
	inline void Scatter(Int2Type<VOLUME>,index_type o,Real const rr[3],TW const &v , TV *cache)const
	{
		Real r[3]
		{	rr[0]-0.5,rr[1]-0.5,rr[2]-0.5};

		int num_of_comp=num_comps_per_cell_[VOLUME];
		int m=0;
		o=SearchCellFix(o,r);
		DEF_INTERPOLATION_SCHEME(,+=v)
	}

	template<typename TV,typename TW>
	inline void Gather(Int2Type<VOLUME>,index_type o,Real const rr[3], TV const*cache,TW *v )const
	{
		Real r[3]
		{	rr[0]-0.5,rr[1]-0.5,rr[2]-0.5};

		int num_of_comp=num_comps_per_cell_[VOLUME];
		int m=0;
		(*v) = 0;
		o=SearchCellFix(o,r);
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

	template<int IF, typename TL, typename ...TI> inline auto
	mapto(Int2Type<IF>, Field<Geometry<this_type, IF>, TL> const &l, TI ... s) const
	DECL_RET_TYPE ((get(l,s...)))

	template<typename TL, typename ...IDXS> inline auto
	mapto(Int2Type<EDGE>, Field<Geometry<this_type, VERTEX>, TL> const &l, int m, IDXS ... s) const
	DECL_RET_TYPE( ((get(l,0,Shift(INC(m),s...)) + get(l,0,s...))*0.5) )

	template<typename TL, typename ...IDXS> inline auto//
	mapto(Int2Type<FACE>, Field<Geometry<this_type, VERTEX>, TL> const &l, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(
					get(l,0,s...)+
					get(l,0,Shift(INC(m+1),s...))+
					get(l,0,Shift(INC(m+2),s...))+
					get(l,0,Shift(INC(m+1) | INC(m+2) ,s...))
			)*0.25

	))
	template<typename TL, typename ...IDXS> inline auto//
	mapto(Int2Type<VOLUME>, Field<Geometry<this_type, VERTEX>, TL> const &l, int m, IDXS ...s) const
	DECL_RET_TYPE(( (
					get(l,0,s...)+
					get(l,0,Shift(X,s...))+
					get(l,0,Shift(Y,s...))+
					get(l,0,Shift(Z,s...))+

					get(l,0,Shift(X|Y,s...))+
					get(l,0,Shift(Z|X,s...))+
					get(l,0,Shift(Z|Y,s...))+
					get(l,0,Shift(Z|X|Y,s...))

			)*0.125

	))

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<VERTEX>, Field<Geometry<this_type, EDGE>, TL> const &l, int m,TI ...s) const
	DECL_RET_TYPE( (get(l,m,s...)+get(l,m,Shift(DES(m),s...)))*0.5 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<FACE>, Field<Geometry<this_type, EDGE>, TL> const &l,int m, TI ...s) const
	DECL_RET_TYPE( (get(l,m,s...)+
			get(l,m,Shift(INC(m+1),s...))+
			get(l,m,Shift(INC(m+2),s...))+
			get(l,m,Shift(INC(m+1)|INC(m+2),s...))+

			get(l,m,Shift(DES(m),s...))+
			get(l,m,Shift(DES(m)|INC(m+1),s...))+
			get(l,m,Shift(DES(m)|INC(m+2),s...))+
			get(l,m,Shift(DES(m)|INC(m+1)|INC(m+2),s...))
	)*0.125 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<VOLUME>, Field<Geometry<this_type, EDGE>, TL> const &l,int m, TI ... s) const
	DECL_RET_TYPE( (get(l,m,s...)+
			get(l,m,Shift(INC(m+1),s...))+
			get(l,m,Shift(INC(m+2),s...))+
			get(l,m,Shift(INC(m+1)|INC(m+2),s...))
	)*0.25 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<VERTEX>, Field<Geometry<this_type, FACE>, TL> const &l, int m,TI ... s) const
	DECL_RET_TYPE( (get(l,m,s...)+
			get(l,m,Shift(DES(m+1),s...))+
			get(l,m,Shift(DES(m+2),s...))+
			get(l,m,Shift(DES(m+1)|DES(m+2),s...))
	)*0.25 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<EDGE>, Field<Geometry<this_type, FACE>, TL> const &l, int m,TI ... s) const
	DECL_RET_TYPE( (
			get(l,m,s...)+
			get(l,m,Shift(DES(m+1),s...))+
			get(l,m,Shift(DES(m+2),s...))+
			get(l,m,Shift(DES(m+1)|DES(m+2),s...))+

			get(l,m,Shift(INC(m),s...))+
			get(l,m,Shift(INC(m)|DES(m+1),s...))+
			get(l,m,Shift(INC(m)|DES(m+2),s...))+
			get(l,m,Shift(INC(m)|DES(m+1)|DES(m+2),s...))
	)*0.125 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<VOLUME>, Field<Geometry<this_type, FACE>, TL> const &l,int m, TI ... s) const
	DECL_RET_TYPE( (get(l,m,s...)+ get(l,m,Shift(INC(m),s...)) )*0.5 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<VERTEX>, Field<Geometry<this_type, VOLUME>, TL> const &l, int m,TI ...s) const
	DECL_RET_TYPE(
	(
			get(l,m,s...)+
			get(l,m,Shift(DES(0),s...))+
			get(l,m,Shift(DES(1),s...))+
			get(l,m,Shift(DES(0)|DES(1),s...))+

			get(l,m,Shift(DES(2),s...))+
			get(l,m,Shift(DES(2)|DES(0),s...))+
			get(l,m,Shift(DES(2)|DES(1),s...))+
			get(l,m,Shift(DES(2)|DES(0)|DES(1),s...))
	)*0.125 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<EDGE>, Field<Geometry<this_type, VOLUME>, TL> const &l, int m,TI ...s) const
	DECL_RET_TYPE(
	(
			get(l,m,s...)+
			get(l,m,Shift(DES(m+1),s...))+
			get(l,m,Shift(DES(m+2),s...))+
			get(l,m,Shift(DES(m+1)|DES(m+2),s...))

	)*0.25 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<FACE>, Field<Geometry<this_type, VOLUME>, TL> const &l,int m, TI ... s) const
	DECL_RET_TYPE( (get(l,m,s...)+ get(l,m,Shift(DES(m),s...)) )*0.5 )

//-----------------------------------------
// Vector Arithmetic
//-----------------------------------------

	template<typename TExpr, typename ... IDXS> inline auto
	OpEval(Int2Type<GRAD>, Field<Geometry<this_type, VERTEX>, TExpr> const & f, int m, IDXS ... s) const
	DECL_RET_TYPE( ( get(f,0,Shift(INC(m),s...))* dS_[0][m] ) +get(f,0,s...)* dS_[1][m])

	template<typename TExpr, typename ...IDX> inline auto
	OpEval(Int2Type<DIVERGE>,Field<Geometry<this_type, EDGE>, TExpr> const & f, int m, IDX ...s) const
	DECL_RET_TYPE((

			(get(f,0,s...)* dS_[0][0] + get(f,0,Shift( NX,s...))* dS_[1][0]) +

			(get(f,1,s...) * dS_[0][1] + get(f,1,Shift( NY,s...))* dS_[1][1]) +

			(get(f,2,s...) * dS_[0][2] + get(f,2,Shift( NZ,s...))* dS_[1][2])
	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURL>,
	Field<Geometry<this_type, EDGE>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			get(f,(m+2)%3,Shift(INC(m+1) ,s...)) * dS_[0][(m + 1) % 3]

			+ get(f,(m+2)%3,s...)* dS_[1][(m + 1) % 3]

			- get(f,(m+1)%3,Shift(INC(m+2) ,s...)) * dS_[0][(m + 2) % 3]

			- get(f,(m+1)%3,s...)* dS_[1][(m + 2) % 3]
	)
	)

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURL>,
	Field<Geometry<this_type, FACE>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			get(f,(m+2)%3,s...)* dS_[0][(m + 1) % 3]

			+ get(f,(m+2)%3,Shift(DES(m+1),s...)) * dS_[1][(m + 1) % 3]

			- get(f,(m+1)%3,s...)* dS_[0][(m + 2) % 3]

			- get(f,(m+1)%3,Shift(DES(m+2),s...)) * dS_[1][(m + 2) % 3]

	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDX>,
	Field<Geometry<this_type, EDGE>, TL> const & f, int m, IDXS ...s) const
//	->typename Field<Geometry<this_type, EDGE>, TL>::value_type
//	{
//		int mm=(m==0?0:(m==1?2:1));
//		typename Field<Geometry<this_type, EDGE>, TL>::value_type res=0;
//		if(m==1)
//		{
//			res=-( get(f,mm,Shift(X,s...)) * dS_[0][0] + get(f,mm,s...)* dS_[1][0]);
//		}
//		else if(m==2)
//
//		{
//			res=( get(f,mm,Shift(X,s...)) * dS_[0][0] + get(f,mm,s...)* dS_[1][0]);
//		}
//		CHECK(res)<<" "<<m;
//		CHECK( dS_[0][0]);
//		CHECK( dS_[1][0]);
//		CHECK( get(f,mm,Shift(X,s...)));
//		CHECK( get(f,mm,s...));
//		return res;
//
//	}
	DECL_RET_TYPE((
			(get(f,(m==0?0:(m==1?2:1)),Shift(X,s...)) * dS_[0][0]
					+ get(f,(m==0?0:(m==1?2:1)),s...)* dS_[1][0])*(m==0?0:(m==1?-1:1))
	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDY>,
	Field<Geometry<this_type, EDGE>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(get(f,(m==1?0:(m==2?0:2)),Shift(Y,s...)) * dS_[0][1]
					+ get(f,(m==1?0:(m==2?0:2)),s...)* dS_[1][1])*(m==1?0:(m==2?-1:1))
	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDZ>,
	Field<Geometry<this_type, EDGE>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(get(f,(m==2?0:(m==0?1:0)),Shift(Z,s...)) * dS_[0][2]
					+ get(f,(m==2?0:(m==0?1:0)),s...)* dS_[1][2])*(m==2?0:(m==0?-1:1))
	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDX>,
	Field<Geometry<this_type, FACE>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(get(f,(m==0?0:(m==1?2:1)),s...) * dS_[0][0]
					+ get(f,(m==0?0:(m==1?2:1)),Shift(NX,s...))* dS_[1][0])*(m==0?0:(m==1?-1:1))
	))
	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDY>,
	Field<Geometry<this_type, FACE>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(get(f,(m==1?0:(m==2?0:2)),s...) * dS_[0][1]
					+ get(f,(m==1?0:(m==2?0:2)),Shift(NY,s...))* dS_[1][1])*(m==1?0:(m==2?-1:1))
	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDZ>,
	Field<Geometry<this_type, FACE>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(get(f,(m==2?0:(m==0?1:0)),s...) * dS_[0][2]
					+ get(f,(m==2?0:(m==0?1:0)),Shift(NZ,s...))
					* dS_[1][2])*(m==2?0:(m==0?-1:1))
	))

	template<int N, typename TL, typename ... IDXS> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,
	Field<Geometry<this_type, N>, TL> const & f, int m, IDXS ... s)
	DECL_RET_TYPE((get(f,m,s...)*dS_[m]))

	template<int IL, int IR, typename TL, typename TR, typename ...TI> inline auto OpEval(Int2Type<WEDGE>,
	Field<Geometry<this_type, IL>, TL> const &l, Field<Geometry<this_type, IR>, TR> const &r, TI ... s) const
	DECL_RET_TYPE( ( mapto(Int2Type<IL+IR>(),l,s...)*
			mapto(Int2Type<IL+IR>(),r,s...)))

	template<int IL, typename TL, typename ...TI> inline auto OpEval(Int2Type<HODGESTAR>,
	Field<Geometry<this_type, IL>, TL> const & f, TI ... s) const
	DECL_RET_TYPE(( mapto(Int2Type<this_type::NUM_OF_DIMS-IL >(),f,s...)))

	/**
	 * non-standard Vector Field operator
	 *
	 */
	template< typename TL, typename TR, typename ...TI> inline auto OpEval(Int2Type<DOT>,
	Field<Geometry<this_type, EDGE>, TL> const &l, Field<Geometry<this_type, EDGE>, TR> const &r,int m, TI ... s) const
	DECL_RET_TYPE( (
			mapto(Int2Type<VERTEX>(),l,0,s...)* mapto(Int2Type<VERTEX>(),r,0,s...)+
			mapto(Int2Type<VERTEX>(),l,1,s...)* mapto(Int2Type<VERTEX>(),r,1,s...)+
			mapto(Int2Type<VERTEX>(),l,2,s...)* mapto(Int2Type<VERTEX>(),r,2,s...)
	))

	template< typename TL, typename TR, typename ...TI> inline auto OpEval(Int2Type<DOT>,
	Field<Geometry<this_type, FACE>, TL> const &l, Field<Geometry<this_type, FACE>, TR> const &r,int m, TI ... s) const
	DECL_RET_TYPE( (
			mapto(Int2Type<VERTEX>(),l,0,s...)* mapto(Int2Type<VERTEX>(),r,0,s...)+
			mapto(Int2Type<VERTEX>(),l,1,s...)* mapto(Int2Type<VERTEX>(),r,1,s...)+
			mapto(Int2Type<VERTEX>(),l,2,s...)* mapto(Int2Type<VERTEX>(),r,2,s...)
	))

	template< int IF,typename TL, typename ...TI> inline auto OpEval(Int2Type<MAPTO0>,
	Field<Geometry<this_type, IF>, TL> const &l, int m, TI ... s) const
	DECL_RET_TYPE( (
			nTuple<3,typename Field<Geometry<this_type, IF>,TL>::value_type>( mapto(Int2Type<VERTEX>(),l,m,s...),
					mapto(Int2Type<VERTEX>(),l,m,s...),
					mapto(Int2Type<VERTEX>(),l,m,s...))
	))

}
;
//******************************************************************************************************
// Offline define
//******************************************************************************************************

template<typename TS>
CoRectMesh<TS>::CoRectMesh(int array_order) :
		array_order_(array_order)
{
}
template<typename TS>
CoRectMesh<TS>::~CoRectMesh()
{
}

inline void _SetImaginaryPart(Real i, Real * v)
{
}

inline void _SetImaginaryPart(Real i, Complex * v)
{
	v->imag(i);
}
template<typename TS>
void CoRectMesh<TS>::Update()
{
	// initialize
	constants();

	tags();

	//configure

	num_cells_ = 1;
	num_grid_points_ = 1;
	cell_volume_ = 1.0;
	d_cell_volume_ = 1.0;
	for (int i = 0; i < NUM_OF_DIMS; ++i)
	{
		if (dims_[i] <= 1)
		{
			dims_[i] = 1;
			dx_[i] = 0.0;
			inv_dx_[i] = 0.0;

			dS_[0][i] = 0.0;

			_SetImaginaryPart(xmax_[i] == xmin_[i] ? 0 : 1.0 / (xmax_[i] - xmin_[i]), &dS_[0][i]);

			dS_[1][i] = 0.0;

			k_[i] = TWOPI * dS_[0][i];

			dims_[i] = 1;

			ghost_width_[i] = 0;

		}
		else
		{
			dx_[i] = (xmax_[i] - xmin_[i]) / static_cast<Real>(dims_[i]);

			inv_dx_[i] = 1.0 / dx_[i];

			dS_[0][i] = 1.0 / dx_[i];

			dS_[1][i] = -1.0 / dx_[i];

			num_cells_ *= (dims_[i]);

			num_grid_points_ *= dims_[i];

			k_[i] = 0.0;

			cell_volume_ *= dx_[i];

			d_cell_volume_ /= dx_[i];

		}
	}

	// Fast first
	if (array_order_ == FORTRAN_ORDER)
	{
		strides_[2] = dims_[1] * dims_[0];
		strides_[1] = dims_[0];
		strides_[0] = 1;
	}
	else
	{
		strides_[2] = 1;
		strides_[1] = dims_[2];
		strides_[0] = dims_[1] * dims_[2];
	}
	for (int i = 0; i < NUM_OF_DIMS; ++i)
	{
		if (dims_[i] <= 1)
			strides_[i] = 0;
	}

	coordinates_shift_[0][0][0] = 0.0;
	coordinates_shift_[0][0][1] = 0.0;
	coordinates_shift_[0][0][2] = 0.0;

	coordinates_shift_[3][0][0] = 0.0;
	coordinates_shift_[3][0][1] = 0.0;
	coordinates_shift_[3][0][2] = 0.0;

	coordinates_shift_[1][0][0] = 0.5 * dx_[0];
	coordinates_shift_[1][0][1] = 0.0;
	coordinates_shift_[1][0][2] = 0.0;

	coordinates_shift_[1][1][0] = 0.0;
	coordinates_shift_[1][1][1] = 0.5 * dx_[1];
	coordinates_shift_[1][1][2] = 0.0;

	coordinates_shift_[1][2][0] = 0.0;
	coordinates_shift_[1][2][1] = 0.0;
	coordinates_shift_[1][2][2] = 0.5 * dx_[2];

	coordinates_shift_[2][0][0] = 0.0;
	coordinates_shift_[2][0][1] = 0.5 * dx_[1];
	coordinates_shift_[2][0][2] = 0.5 * dx_[2];

	coordinates_shift_[2][1][0] = 0.5 * dx_[0];
	coordinates_shift_[2][1][1] = 0.0;
	coordinates_shift_[2][1][2] = 0.5 * dx_[2];

	coordinates_shift_[2][2][0] = 0.5 * dx_[0];
	coordinates_shift_[2][2][1] = 0.5 * dx_[1];
	coordinates_shift_[2][2][2] = 0.0;

}

template<typename TS>
template<typename ISTREAM> inline void CoRectMesh<TS>::Deserialize(ISTREAM const &cfg)
{

	{
		if (cfg.empty())
		{
			WARNING << "Empty configure!";
			return;
		}
		if (cfg["Type"].template as<std::string>("Real") != GetTypeName())
		{
			WARNING << "illegal config [Type: except=" << GetTypeName() << ", configure="
					<< cfg["Type"].template as<std::string>() << "]";

			return;
		}

		auto cfg_scalar_type = cfg["ScalarType"].template as<std::string>();

		auto this_scalar_type = ((std::is_same<TS, Complex>::value) ? "Complex" : "Real");

		if (cfg_scalar_type != "" && cfg_scalar_type != this_scalar_type)
		{
			WARNING << "illegal configure[Scalar Type: except= " << this_scalar_type << ", configure="
					<< cfg_scalar_type << "]";
		}

	}

	LOGGER << "Deserialize CoRectMesh" << START;

	constants().Deserialize(cfg["UnitSystem"]);

	tags().Deserialize(cfg["MediaTag"]);

	auto topology = cfg["Topology"];
	topology["Dimensions"].as(&dims_);

	auto geometry = cfg["Geometry"];

	geometry["Min"].as(&xmin_);
	geometry["Max"].as(&xmax_);
	geometry["dt"].as(&dt_);

	Update();

	LOGGER << "Deserialize CoRectMesh [" << GetTypeName() << "] " << DONE;
}
template<typename TS>
template<typename OSTREAM> inline OSTREAM &
CoRectMesh<TS>::Serialize(OSTREAM &os) const
{

	os

	<< "{" << "\n"

	<< std::setw(10) << "Type" << " = \"" << GetTypeName() << "\", \n"

	<< std::setw(10) << "ScalarType" << " = \""

	<< ((std::is_same<TS, Complex>::value) ? "Complex" : "Real") << "\", \n"

	<< std::setw(10) << "Topology" << " = { \n "

	<< "\t" << std::setw(10) << "Type" << " = \"" << GetTopologyTypeAsString() << "\", \n"

	<< "\t" << std::setw(10) << "Dimensions" << " = {" << ToString(dims_, ",") << "}, \n "

	<< "\t" << std::setw(10) << "GhostWidth" << " = {" << ToString(ghost_width_, ",") << "}, \n "

	<< std::setw(10) << "}, \n "

	<< std::setw(10) << "Geometry" << " = { \n "

	<< "\t" << std::setw(10) << "Type" << " = \"Origin_DxDyDz\", \n "

	<< "\t" << std::setw(10) << "Origin" << " = {" << ToString(xmin_, ",") << "}, \n "

	<< "\t" << std::setw(10) << "DxDyDz" << " = {" << ToString(dx_, ",") << "}, \n "

	<< "\t" << std::setw(10) << "Min" << " = {" << ToString(xmin_, ",") << "}, \n "

	<< "\t" << std::setw(10) << "Max" << " = {" << ToString(xmax_, ",") << "}, \n "

	<< "\t" << std::setw(10) << "k" << " = {" << ToString(k_, ",") << "}, \n "

	<< std::setw(10) << "}, \n "

	<< std::setw(10) << "dt" << " = " << GetDt() << ",\n"

	<< std::setw(10) << "Unit System" << " = " << constants() << ",\n"

	<< std::setw(10) << "Media Tags" << " = " << tags() << "\n"

	<< "} \n ";

	return os;
}

template<typename TS> inline std::ostream &
operator<<(std::ostream & os, CoRectMesh<TS> const & d)
{
	d.Serialize(os);
	return os;
}

template<typename TS>
void CoRectMesh<TS>::_Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
		std::function<void(int, index_type, index_type, index_type)> const &fun) const
{

//	index_type ib = ((flags & WITH_GHOSTS) > 0) ? 0 : ghost_width_[0];
//	index_type ie = ((flags & WITH_GHOSTS) > 0) ? dims_[0] : dims_[0] - ghost_width_[0];
//
//	index_type jb = ((flags & WITH_GHOSTS) > 0) ? 0 : ghost_width_[1];
//	index_type je = ((flags & WITH_GHOSTS) > 0) ? dims_[1] : dims_[1] - ghost_width_[1];
//
//	index_type kb = ((flags & WITH_GHOSTS) > 0) ? 0 : ghost_width_[2];
//	index_type ke = ((flags & WITH_GHOSTS) > 0) ? dims_[2] : dims_[2] - ghost_width_[2];

	index_type ib = 0;
	index_type ie = dims_[0];

	index_type jb = 0;
	index_type je = dims_[1];

	index_type kb = 0;
	index_type ke = dims_[2];

	int mb = 0;
	int me = num_comps_per_cell_[IFORM];

	index_type len = ie - ib;
	index_type tb = ib + len * thread_id / num_threads;
	index_type te = ib + len * (thread_id + 1) / num_threads;

	for (index_type i = tb; i < te; ++i)
		for (index_type j = jb; j < je; ++j)
			for (index_type k = kb; k < ke; ++k)
				for (int m = mb; m < me; ++m)
				{
					fun(m, i, j, k);
				}

}

template<typename TS>
template<typename ...Args>
void CoRectMesh<TS>::ParallelTraversal(Args const &...args) const
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

template<typename TS>
template<typename ...Args>
void CoRectMesh<TS>::SerialTraversal(Args const &...args) const
{
	_Traversal(1, 0, std::forward<Args const&>( args)...);
}

}
// namespace simpla

#endif //UNIFORM_RECT_H_
