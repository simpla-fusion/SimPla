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
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <thread>
#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../physics/physical_constants.h"
#include "../physics/constants.h"
#include "../utilities/memory_pool.h"
#include "../utilities/log.h"
#include "field_convert.h"
namespace simpla
{

/**
 *
 *  @brief UniformRectMesh -- Uniform rectangular structured grid.
 *  @ingroup mesh 
 * */

template<typename TS = Real>
struct CoRectMesh
{
	typedef CoRectMesh this_type;

	static constexpr unsigned int MAX_NUM_NEIGHBOUR_ELEMENT = 12;
	static constexpr unsigned int MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr unsigned int NUM_OF_DIMS = 3;
	static constexpr unsigned int NUM_OF_COMPONENT_TYPE = NUM_OF_DIMS + 1;

	typedef size_t index_type;

	typedef TS scalar;

	typedef TS scalar_type;

	typedef nTuple<3, Real> coordinates_type;

	typedef unsigned int tag_type;

	PhysicalConstants constants;

	this_type & operator=(const this_type&) = delete;

	Real dt_ = 0.0;

	template<typename U>
	friend std::ostream & operator<<(std::ostream &os, CoRectMesh<U> const &);

	// Topology
	unsigned int DEFAULT_GHOST_WIDTH = 2;

	nTuple<NUM_OF_DIMS, size_t> shift_ = { 0, 0, 0 };
	nTuple<NUM_OF_DIMS, size_t> dims_ = { 11, 11, 11 };
	nTuple<NUM_OF_DIMS, size_t> ghost_width_ = { DEFAULT_GHOST_WIDTH, DEFAULT_GHOST_WIDTH, DEFAULT_GHOST_WIDTH };
	nTuple<NUM_OF_DIMS, size_t> strides_ = { 0, 0, 0 };

	size_t num_cells_ = 0;

	size_t num_grid_points_ = 0;

	// Geometry
	coordinates_type xmin_ = { 0, 0, 0 };
	coordinates_type xmax_ = { 10, 10, 10 };

	nTuple<NUM_OF_DIMS, scalar> dS_[2] = { 0, 0, 0, 0, 0, 0 };
	nTuple<NUM_OF_DIMS, scalar> k_ = { 0, 0, 0 };

	coordinates_type dx_ = { 0, 0, 0 };
	coordinates_type inv_dx_ = { 0, 0, 0 };

	Real cell_volume_ = 1.0;
	Real d_cell_volume_ = 1.0;

	const int num_comps_per_cell_[NUM_OF_COMPONENT_TYPE] = { 1, 3, 3, 1 };

	coordinates_type coordinates_shift_[NUM_OF_COMPONENT_TYPE][NUM_OF_DIMS];

	CoRectMesh()
	{
	}

	~CoRectMesh()
	{
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

	template<typename TV> using Container=std::shared_ptr<TV>;

	template<int iform, typename TV> inline Container<TV> MakeContainer() const
	{
		return (MEMPOOL.allocate_shared_ptr<TV>(GetNumOfElements(iform)));
	}

	template<typename ISTREAM> void Deserialize(ISTREAM const &vm);

	template<typename OSTREAM> OSTREAM& Serialize(OSTREAM &vm) const;

	inline void _SetImaginaryPart(Real i, Real * v)
	{
	}

	inline void _SetImaginaryPart(Real i, Complex * v)
	{
		v->imag(i);
	}
	void Update()
	{
		num_cells_ = 1;
		num_grid_points_ = 1;
		cell_volume_=1.0;
		d_cell_volume_=1.0;
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

				ghost_width_[i]=0;

			}
			else
			{
				dx_[i] = (xmax_[i] - xmin_[i]) / static_cast<Real>(dims_[i] - 1);

				inv_dx_[i]=1.0/dx_[i];

				dS_[0][i] = 1.0 / dx_[i];

				dS_[1][i] = -1.0 / dx_[i];

				num_cells_ *= (dims_[i] - 1);

				num_grid_points_ *= dims_[i];

				k_[i] = 0.0;

			}

			if (!isinf(dx_[i]))
			{
				cell_volume_ *= (dims_[i] - 1) * dx_[i];
			}

			if (!isinf(dx_[i]) && dx_[i] > 0)
			{
				d_cell_volume_ *= dx_[i];
			}

		}

		strides_[2] = 1;
		strides_[1] = dims_[2];
		strides_[0] = dims_[1] * dims_[2];

		for(int i=0;i<NUM_OF_DIMS;++i)
		{
			if(dims_[i]<=1)strides_[i]=0;
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

public:

	inline coordinates_type GetCoordinates(int IFORM, int m, index_type i, index_type j, index_type k) const
	{

		coordinates_type res = xmin_;
		res[0] += i * dx_[0] + coordinates_shift_[IFORM][m][0];
		res[1] += i * dx_[1] + coordinates_shift_[IFORM][m][1];
		res[2] += i * dx_[2] + coordinates_shift_[IFORM][m][2];
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
	inline size_t GetSubComponent(size_t s) const
	{
		return s % num_comps_per_cell_[IFORM];
	}
	template<typename ... IDXS>
	inline size_t GetComponentIndex(int IFORM, int m, IDXS ... s) const
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
		return v[GetSubComponent<IFORM>(s)];
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
	inline int _GetNeighbourCell(Int2Type<1>, Int2Type<0>, index_type *v, int m, Args ... s) const
	{
		v[0] = GetIndex(s...);
		v[1] = Shift(INC(m), s...);
		return 2;
	}

	template<typename ... Args>
	inline int _GetNeighbourCell(Int2Type<2>, Int2Type<0>, index_type *v, int m, Args ... s) const
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
	inline int _GetNeighbourCell(Int2Type<3>, Int2Type<0>, index_type *v, int m, Args ... s) const
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
	inline int _GetNeighbourCell(Int2Type<0>, Int2Type<1>, index_type *v, int m, Args ... s) const
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
	inline int _GetNeighbourCell(Int2Type<2>, Int2Type<1>, index_type *v, int m, Args ... s) const
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
	inline int _GetNeighbourCell(Int2Type<3>, Int2Type<1>, index_type *v, int m, Args ... s) const
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
	inline int _GetNeighbourCell(Int2Type<0>, Int2Type<2>, index_type *v, int m, Args ... s) const
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
	inline int _GetNeighbourCell(Int2Type<1>, Int2Type<2>, index_type *v, int m, Args ... s) const
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
	inline int _GetNeighbourCell(Int2Type<3>, Int2Type<2>, index_type *v, int m, Args ... s) const
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
	inline int _GetNeighbourCell(Int2Type<0>, Int2Type<3>, index_type *v, int m, Args ... s) const
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
	inline int _GetNeighbourCell(Int2Type<1>, Int2Type<3>, index_type *v, int m, Args ... s) const
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
	inline int _GetNeighbourCell(Int2Type<2>, Int2Type<3>, index_type *v, int m, Args ... s) const
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

	enum
	{
		NIL = 0, // 00 00 00
		X = 1,// 00 00 01
		NX = 2,// 00 00 10
		Y = 4,// 00 01 00
		NY = 8,// 00 10 00
		Z = 16,// 01 00 00
		NZ = 32// 10 00 00
	};
	inline size_t INC(int m) const
	{
		return 1 << (m % 3) * 2;
	}
	inline size_t DES(int m) const
	{
		return 2 << (m % 3) * 2;
	}

	void UnpackIndex(index_type *idx,size_t s)const
	{
		UnpackIndex(idx,idx+1,idx+2, s);
	}
	void UnpackIndex(index_type *i,index_type *j,index_type *k,size_t s)const
	{
		*i =(strides_[0]==0)?0:(s/strides_[0]);
		s-=(*i)*strides_[0];
		*j =(strides_[1]==0)?0:(s/strides_[1]);
		s-=(*j)*strides_[1];
		*k =s;
	}
//	void UnpackIndex(index_type *i,index_type *j,index_type *k,index_type i1,index_type j1,index_type k1 )
//	{
//		*i=i1;
//		*j=j1;
//		*k=k1;
//	}

	/**
	 * (((d & 3) + 1) % 3 - 1)
	 *
	 * 00 -> 0
	 * 01 -> 1
	 * 10 -> -1
	 *
	 */
	template<typename ... IDXS>
	inline size_t Shift(int d, IDXS ... s) const
	{
		index_type i,j,k;
		UnpackIndex(&i,&j,&k,s...);
		return Shift(d,i,j,k);

//		return GetIndex(s...)
//
//		+ ((((d >> 4) & 3) + 1) % 3 - 1) * strides_[2]
//
//		+ ((((d >> 2) & 3) + 1) % 3 - 1) * strides_[1]
//
//		+ (((d & 3) + 1) % 3 - 1) * strides_[0];

	}

	inline size_t Shift(int d, index_type i, index_type j, index_type k) const
	{
		return

		(((i + (((d & 3) + 1) % 3 - 1)) % dims_[0]) * strides_[0]

		+ ((j + ((((d >> 2) & 3) + 1) % 3 - 1)) % dims_[1]) * strides_[1]

		+ ((k + ((((d >> 4) & 3) + 1) % 3 - 1)) % dims_[2]) * strides_[2]);
	}

	inline size_t GetIndex(index_type i, index_type j, index_type k) const
	{
		return ((i % dims_[0]) * strides_[0] + (j % dims_[1]) * strides_[1] + (k % dims_[2]) * strides_[2]);
	}
	inline size_t GetIndex(index_type s) const
	{
		return s;
	}
	template<typename T, typename ... TI>
	inline typename std::enable_if<!is_field<T>::value, T>::type get(T const &l, TI ...) const
	{
		return std::move(l);
	}

	template<int IFORM, typename TL> inline typename Field<Geometry<this_type, IFORM>, TL>::value_type & get(
	Field<Geometry<this_type, IFORM>, TL> *l, size_t s) const
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
	TV & get_value(Container<TV> & d,size_t s)const
	{
		return * (d.get()+s);
	}
	template<typename TV>
	TV const & get_value(Container<TV> const& d,size_t s)const
	{
		return * (d.get()+s);
	}

/// Traversal
	enum
	{
		WITH_GHOSTS = 1
	};

	bool default_parallel_=true;

	template<typename ... Args>
	void Traversal(Args const &...args) const
	{
		if (default_parallel_)
		{
			ParallelTraversal(std::forward<Args const &>(args)... );
		}
		else
		{
			SerialTraversal(std::forward<Args const &>(args)... );
		}
	}

	template<typename ...Args> void ParallelTraversal(Args const &...args)const;

	template<typename ...Args> void SerialTraversal(Args const &...args)const;

	void _Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
	std::function<void(int, index_type, index_type, index_type)> const &fun, unsigned int flags=0) const;

	void _Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
	std::function<void(index_type)> const &fun, unsigned int flag = 0) const
	{
		_Traversal(num_threads,thread_id,
		IFORM, [&](int m,index_type i,index_type j,index_type k)
		{
			fun(GetComponentIndex(IFORM,m,i,j,k));
		}, flag);

	}
	void _Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
	std::function<void(index_type, coordinates_type)> const &fun, unsigned int flag = 0) const
	{
		int num = num_comps_per_cell_[IFORM];

		_Traversal(num_threads,thread_id,
		IFORM, [&](int m,index_type i,index_type j,index_type k)
		{
			fun(GetComponentIndex(IFORM,m,i,j,k),this->GetCoordinates(IFORM,m,i,j,k));
		}, flag);

	}

public:

	template<typename TFUN>
	inline void TraversalSubComponent(int IFORM, index_type s, TFUN const & fun) const
	{
		int num = num_comps_per_cell_[IFORM];
		for (int i = 0; i < num; ++i)
		{
			fun(s * num + i);
		}
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

//	template<typename Fun, typename TF, typename ... Args> inline
//	void ForAll(unsigned int flag, Fun const &fun, TF const & l, Args const& ... args) const
//	{
//		Traversal(FieldTraits<TF>::IForm, [&](int m,index_type i,index_type j,index_type k)
//		{	fun(get(l,m,i,j,k),get(args,m,i,j,k)...);}, flag);
//	}
//
//	template<typename Fun, typename TF, typename ...Args> inline
//	void ForAll(unsigned int flag, Fun const &fun, TF * l, Args const & ... args) const
//	{
//		Traversal(FieldTraits<TF>::IForm, [&](int m,index_type i,index_type j,index_type k)
//		{	fun(get(l,m,i,j,k),get(args,m,i,j,k)...);}, flag);
//	}

//	template<typename Fun, typename TF, typename ... Args> inline
//	void ForEach(Fun const &fun, TF & l, Args const& ... args, unsigned int flag = 0) const
//	{
//		Traversal(FieldTraits<TF>::IForm, [&](int m,index_type i,index_type j,index_type k)
//		{	fun(get(l,m,i,j,k),get(args,m,i,j,k)...);}, 0);
//	}
//
//	template<typename Fun, typename TF, typename ...Args> inline
//	void ForEach(Fun const &fun, TF * l, Args const & ... args) const
//	{
//		Traversal(FieldTraits<TF>::IForm, [&](int m,index_type i,index_type j,index_type k)
//		{	fun(get(l,m,i,j,k),get(args,m,i,j,k)...);}, 0);
//	}

	template<typename TL, typename TR> void AssignContainer(int IFORM, TL * lhs, TR const &rhs) const
	{
		ParallelTraversal(IFORM, [&](int m, index_type x, index_type y, index_type z)
		{	get(lhs,m,x,y,z)=get(rhs,m,x,y,z);});

	}
	template<typename TL> void AssignContainer(int IFORM, TL * lhs, TL const &rhs) const
	{
		ParallelTraversal(IFORM, [&](int m, index_type x, index_type y, index_type z)
		{	get(lhs,m,x,y,z)=get(rhs,m,x,y,z);},WITH_GHOSTS);

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

		}, WITH_GHOSTS);
	}

	template<typename TL,int IL>
	typename std::enable_if<IL==0|IL==3,void>::type AssignContainer(Field<Geometry<this_type,IL> ,TL> * lhs,
	typename Field<Geometry<this_type,IL> ,TL>::field_value_type const &rhs) const
	{
		ParallelTraversal(0, [&](int m, index_type x, index_type y, index_type z)
		{	get(lhs,0,x,y,z)=rhs;}, WITH_GHOSTS);
	}

// Properties of UniformRectMesh --------------------------------------
	inline void SetGhostWidth(int i,size_t v)
	{
		ghost_width_[i% NUM_OF_DIMS]=v;
	}

	inline nTuple<NUM_OF_DIMS,size_t> const&GetGhostWidth( )const
	{
		return ghost_width_;
	}

	inline void SetExtent(coordinates_type const & pmin, coordinates_type const & pmax)
	{
		xmin_ = pmin;
		xmax_ = pmax;

		Update();
	}

	inline std::pair<coordinates_type, coordinates_type> GetExtent() const
	{
		return std::move(std::make_pair(xmin_, xmax_));
	}

	inline void SetDimension(nTuple<NUM_OF_DIMS, size_t> const & pdims)
	{
		dims_ = pdims;

		Update();
	}
	inline nTuple<NUM_OF_DIMS, size_t> const & GetDimension() const
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

	inline Real GetDt() const
	{
		return dt_;
	}

	inline void SetDt(Real dt = 0.0)
	{
		dt_ = dt;
		Update();
	}

	inline size_t GetNumOfElements(int iform) const
	{

		return (num_grid_points_ * num_comps_per_cell_[iform]);
	}

	inline size_t GetNumOfVertices(...) const
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

	/**
	 * Locate the cell containing a specified point.
	 * @param x
	 * @param pcoords local parameter coordinates
	 * @return index of cell
	 */
	inline index_type SearchCell(coordinates_type const &x, Real *pcoords = nullptr) const
	{

		size_t idx = 0;

		index_type i,j,k;

		Real e[3]=
		{	0,0,0};

		i= (dims_[0]<=1)?0:static_cast<size_t>(std::modf((x[0] - xmin_[0]) * inv_dx_[0],e));
		j= (dims_[1]<=1)?0:static_cast<size_t>(std::modf((x[1] - xmin_[1]) * inv_dx_[1],e));
		k= (dims_[2]<=1)?0:static_cast<size_t>(std::modf((x[2] - xmin_[2]) * inv_dx_[2],e));

		if (pcoords != nullptr)
		{
			pcoords[0] = e[0];
			pcoords[1] = e[1];
			pcoords[2] = e[2];
		}

		return GetIndex(i,j,k);
	}
	/**
	 *  Speed up version SearchCell, restain for curvline or unstructured grid
	 * @param
	 * @param x
	 * @param pcoords
	 * @return index of cell
	 */
	inline index_type SearchCell(index_type const &hint_idx, coordinates_type const &x, Real *pcoords = nullptr) const
	{
		return SearchCell(x, pcoords);
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

	inline int GetAffectedPoints(Int2Type<0>, index_type const & s=0, size_t * points=nullptr, int affect_region = 1) const
	{

		if(points!=nullptr)
		{
			index_type i,j,k;
			UnpackIndex(&i,&j,&k,s);
			points[0] = Shift(0,i,j,k);
			points[1] = Shift(X,i,j,k);
			points[2] = Shift(Y,i,j,k);
			points[3] = Shift(X|Y,i,j,k);
			points[4] = Shift(Z,i,j,k);
			points[5] = Shift(Z|X,i,j,k);
			points[6] = Shift(Z|Y,i,j,k);;
			points[7] = Shift(Z|X|Y,i,j,k);;
		}
		return 8;
	}

	inline int GetAffectedPoints(Int2Type<1>, index_type const & s =0, size_t * points=nullptr, int affect_region = 1) const
	{

		if(points!=nullptr)
		{
			index_type i,j,k;
			UnpackIndex(&i,&j,&k,s);
			points[0] = Shift(0,i,j,k);
			points[1] = Shift(X,i,j,k);
			points[2] = Shift(Y,i,j,k);
			points[3] = Shift(X|Y,i,j,k);
			points[4] = Shift(Z,i,j,k);
			points[5] = Shift(Z|X,i,j,k);
			points[6] = Shift(Z|Y,i,j,k);;
			points[7] = Shift(Z|X|Y,i,j,k);;
			points[8] = Shift(Z,i,j,k);
			points[9] = Shift(Z|X,i,j,k);
			points[10] = Shift(Z|Y,i,j,k);;
			points[11] = Shift(Z|X|Y,i,j,k);;
		}
		return 12;
	}

	inline int GetAffectedPoints(Int2Type<2>, index_type const & idx =0, size_t * points=nullptr, int affect_region = 1) const
	{
		return 6;
	}

	inline int GetAffectedPoints(Int2Type<3>, index_type const & idx =0, size_t * points=nullptr, int affect_region = 1) const
	{
		return 1;
	}

	template<typename TV,typename TW>
	inline void ScatterToMesh(Int2Type<0>,Real const *pcoords, TW const & v,TV* cache, int affect_region = 1) const
	{
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];

		cache[0] += v* (1.0 - r) * (1.0 - s) * (1.0 - t);
		cache[1] += v* r * (1.0 - s) * (1.0 - t);
		cache[2] += v* (1.0 - r) * s * (1.0 - t);
		cache[3] += v* r * s * (1.0 - t);
		cache[4] += v* (1.0 - r) * (1.0 - s) * t;
		cache[5] += v* r * (1.0 - s) * t;
		cache[6] += v* (1.0 - r) * s * t;
		cache[7] += v* r * s * t;
	}

	template<typename TV,typename TW>
	inline void ScatterToMesh(Int2Type<1>,Real const *pcoords, TW const & v,TV* cache, int affect_region = 1) const
	{
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];
	}
	template<typename TV,typename TW>
	inline void ScatterToMesh(Int2Type<2>,Real const *pcoords, TW const & v,TV* cache, int affect_region = 1) const
	{
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];
	}
	template<typename TV,typename TW>
	inline void ScatterToMesh(Int2Type<3>,Real const *pcoords, TW const & v,TV* cache, int affect_region = 1) const
	{
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];
	}

	template<typename TV,typename TW>
	inline void GatherFromMesh(Int2Type<0>, Real const *pcoords, TV const* cache, TW* res, int affect_region = 1) const
	{
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];

		(*res) = 0;
		(*res)+=cache[0] * (1.0 - r) * (1.0 - s) * (1.0 - t);
		(*res)+=cache[1] * r * (1.0 - s) * (1.0 - t);
		(*res)+=cache[2] * (1.0 - r) * s * (1.0 - t);
		(*res)+=cache[3] * r * s * (1.0 - t);
		(*res)+=cache[4] * (1.0 - r) * (1.0 - s) * t;
		(*res)+=cache[5] * r * (1.0 - s) * t;
		(*res)+=cache[6] * (1.0 - r) * s * t;
		(*res)+=cache[7] * r * s * t;
	}

	template<typename TV,typename TW>
	inline void GatherFromMesh(Int2Type<1>, Real const *pcoords, TV const* cache, TW* res, int affect_region = 1) const
	{
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];
	}

	template<typename TV,typename TW>
	inline void GatherFromMesh(Int2Type<2>, Real const *pcoords, TV const* cache, TW* res, int affect_region = 1) const
	{
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];
	}
	template<typename TV,typename TW>
	inline void GatherFromMesh(Int2Type<3>, Real const *pcoords, TV const* cache, TW* res, int affect_region = 1) const
	{
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];
	}
// Mapto ----------------------------------------------------------
	/**
	 *    mapto(Int2Type<0> ,   //tarGet topology position
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
	mapto(Int2Type<1>, Field<Geometry<this_type, 0>, TL> const &l, int m, IDXS ... s) const
	DECL_RET_TYPE( ((get(l,m,Shift(INC(m),s...)) + get(l,m,s...))*0.5) )

	template<typename TL, typename ...IDXS> inline auto//
	mapto(Int2Type<2>, Field<Geometry<this_type, 0>, TL> const &l, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(
					get(l,0,s...)+
					get(l,0,Shift(INC(m+1),s...))+
					get(l,0,Shift(INC(m+2),s...))+
					get(l,0,Shift(INC(m+1) | INC(m+2) ,s...))
			)*0.25

	))
	template<typename TL, typename ...IDXS> inline auto//
	mapto(Int2Type<3>, Field<Geometry<this_type, 0>, TL> const &l, int m, IDXS ...s) const
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
	inline auto mapto(Int2Type<0>, Field<Geometry<this_type, 1>, TL> const &l, int m,TI ...s) const
	DECL_RET_TYPE( (get(l,m,s...)+get(l,m,Shift(DES(m),s...)))*0.5 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<2>, Field<Geometry<this_type, 1>, TL> const &l,int m, TI ...s) const
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
	inline auto mapto(Int2Type<3>, Field<Geometry<this_type, 1>, TL> const &l,int m, TI ... s) const
	DECL_RET_TYPE( (get(l,m,s...)+
			get(l,m,Shift(INC(m+1),s...))+
			get(l,m,Shift(INC(m+2),s...))+
			get(l,m,Shift(INC(m+1)|INC(m+2),s...))
	)*0.25 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<0>, Field<Geometry<this_type, 2>, TL> const &l, int m,TI ... s) const
	DECL_RET_TYPE( (get(l,m,s...)+
			get(l,m,Shift(DES(m+1),s...))+
			get(l,m,Shift(DES(m+2),s...))+
			get(l,m,Shift(DES(m+1)|DES(m+2),s...))
	)*0.25 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<1>, Field<Geometry<this_type, 2>, TL> const &l, int m,TI ... s) const
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
	inline auto mapto(Int2Type<3>, Field<Geometry<this_type, 2>, TL> const &l,int m, TI ... s) const
	DECL_RET_TYPE( (get(l,m,s...)+ get(l,m,Shift(INC(m),s...)) )*0.5 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<0>, Field<Geometry<this_type, 3>, TL> const &l, int m,TI ...s) const
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
	inline auto mapto(Int2Type<1>, Field<Geometry<this_type, 3>, TL> const &l, int m,TI ...s) const
	DECL_RET_TYPE(
	(
			get(l,m,s...)+
			get(l,m,Shift(DES(m+1),s...))+
			get(l,m,Shift(DES(m+2),s...))+
			get(l,m,Shift(DES(m+1)|DES(m+2),s...))

	)*0.25 )

	template<typename TL, typename ...TI>
	inline auto mapto(Int2Type<2>, Field<Geometry<this_type, 3>, TL> const &l,int m, TI ... s) const
	DECL_RET_TYPE( (get(l,m,s...)+ get(l,m,Shift(DES(m),s...)) )*0.5 )

//-----------------------------------------
// Vector Arithmetic
//-----------------------------------------

	template<typename TExpr, typename ... IDXS> inline auto
	OpEval(Int2Type<GRAD>, Field<Geometry<this_type, 0>, TExpr> const & f, int m, IDXS ... s) const
	DECL_RET_TYPE( ( get(f,0,Shift(INC(m),s...))* dS_[0][m] ) +get(f,0,s...)* dS_[1][m])

	template<typename TExpr, typename ...IDX> inline auto
	OpEval(Int2Type<DIVERGE>,Field<Geometry<this_type, 1>, TExpr> const & f, int m, IDX ...s) const
	DECL_RET_TYPE((

			(get(f,0,s...)* dS_[0][0] + get(f,0,Shift( NX,s...))* dS_[1][0]) +

			(get(f,1,s...) * dS_[0][1] + get(f,1,Shift( NY,s...))* dS_[1][1]) +

			(get(f,2,s...) * dS_[0][2] + get(f,2,Shift( NZ,s...))* dS_[1][2])
	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURL>,
	Field<Geometry<this_type, 1>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			get(f,(m+2)%3,Shift(INC(m+1) ,s...)) * dS_[0][(m + 1) % 3]

			+ get(f,(m+2)%3,s...)* dS_[1][(m + 1) % 3]

			- get(f,(m+1)%3,Shift(INC(m+2) ,s...)) * dS_[0][(m + 2) % 3]

			- get(f,(m+1)%3,s...)* dS_[1][(m + 2) % 3]
	)
	)

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURL>,
	Field<Geometry<this_type, 2>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			get(f,(m+2)%3,s...)* dS_[0][(m + 1) % 3]

			+ get(f,(m+2)%3,Shift(DES(m+1),s...)) * dS_[1][(m + 1) % 3]

			- get(f,(m+1)%3,s...)* dS_[0][(m + 2) % 3]

			- get(f,(m+1)%3,Shift(DES(m+2),s...)) * dS_[1][(m + 2) % 3]

	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDX>,
	Field<Geometry<this_type, 1>, TL> const & f, int m, IDXS ...s) const
//	->typename Field<Geometry<this_type, 1>, TL>::value_type
//	{
//		int mm=(m==0?0:(m==1?2:1));
//		typename Field<Geometry<this_type, 1>, TL>::value_type res=0;
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
	Field<Geometry<this_type, 1>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(get(f,(m==1?0:(m==2?0:2)),Shift(Y,s...)) * dS_[0][1]
					+ get(f,(m==1?0:(m==2?0:2)),s...)* dS_[1][1])*(m==1?0:(m==2?-1:1))
	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDZ>,
	Field<Geometry<this_type, 1>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(get(f,(m==2?0:(m==0?1:0)),Shift(Z,s...)) * dS_[0][2]
					+ get(f,(m==2?0:(m==0?1:0)),s...)* dS_[1][2])*(m==2?0:(m==0?-1:1))
	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDX>,
	Field<Geometry<this_type, 2>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(get(f,(m==0?0:(m==1?2:1)),s...) * dS_[0][0]
					+ get(f,(m==0?0:(m==1?2:1)),Shift(NX,s...))* dS_[1][0])*(m==0?0:(m==1?-1:1))
	))
	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDY>,
	Field<Geometry<this_type, 2>, TL> const & f, int m, IDXS ...s) const
	DECL_RET_TYPE((
			(get(f,(m==1?0:(m==2?0:2)),s...) * dS_[0][1]
					+ get(f,(m==1?0:(m==2?0:2)),Shift(NY,s...))* dS_[1][1])*(m==1?0:(m==2?-1:1))
	))

	template<typename TL, typename ...IDXS> inline auto OpEval(Int2Type<CURLPDZ>,
	Field<Geometry<this_type, 2>, TL> const & f, int m, IDXS ...s) const
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
	Field<Geometry<this_type, 1>, TL> const &l, Field<Geometry<this_type, 1>, TR> const &r,int m, TI ... s) const
	DECL_RET_TYPE( (
			mapto(Int2Type<0>(),l,0,s...)* mapto(Int2Type<0>(),r,0,s...)+
			mapto(Int2Type<0>(),l,1,s...)* mapto(Int2Type<0>(),r,1,s...)+
			mapto(Int2Type<0>(),l,2,s...)* mapto(Int2Type<0>(),r,2,s...)
	))

	template< typename TL, typename TR, typename ...TI> inline auto OpEval(Int2Type<DOT>,
	Field<Geometry<this_type, 2>, TL> const &l, Field<Geometry<this_type, 2>, TR> const &r,int m, TI ... s) const
	DECL_RET_TYPE( (
			mapto(Int2Type<0>(),l,0,s...)* mapto(Int2Type<0>(),r,0,s...)+
			mapto(Int2Type<0>(),l,1,s...)* mapto(Int2Type<0>(),r,1,s...)+
			mapto(Int2Type<0>(),l,2,s...)* mapto(Int2Type<0>(),r,2,s...)
	))

	template< int IF,typename TL, typename ...TI> inline auto OpEval(Int2Type<MAPTO0>,
	Field<Geometry<this_type, IF>, TL> const &l, int m, TI ... s) const
	DECL_RET_TYPE( (
			nTuple<3,typename Field<Geometry<this_type, IF>,TL>::value_type>( mapto(Int2Type<0>(),l,m,s...),
					mapto(Int2Type<0>(),l,m,s...),
					mapto(Int2Type<0>(),l,m,s...))
	))

}
;
template<typename TS>
template<typename ISTREAM> inline void CoRectMesh<TS>::Deserialize(ISTREAM const &vm)
{
	constants.Deserialize(vm.GetChild("UnitSystem"));

	vm.GetChild("Topology").template GetValue("Dimensions", &dims_);
	vm.GetChild("Topology").template GetValue("GhostWidth", &ghost_width_);

	vm.GetChild("Geometry").template GetValue("Min", &xmin_);
	vm.GetChild("Geometry").template GetValue("Max", &xmax_);
	vm.GetChild("Geometry").template GetValue("dt", &dt_);

	Update();

	LOGGER << "Load Mesh " << GetTypeName() << DONE;

}
template<typename TS>
template<typename OSTREAM> inline OSTREAM &
CoRectMesh<TS>::Serialize(OSTREAM &os) const
{

	os

	<< "--  Grid " << "\n"

	<< "Grid={" << "\n"

	<< "        Type = \"" << GetTypeName() << "\", \n"

	<< "        ScalarType = \""

	<< ((std::is_same<TS, Complex>::value) ? "Complex" : "Real") << "\", \n"

	<< "	Topology={ \n "

	<< "        Type = \"" << GetTopologyTypeAsString() << "\", \n"

	<< "		Dimensions = {" << ToString(dims_, ",") << "}, \n "

	<< "		GhostWidth = {" << ToString(ghost_width_, ",") << "}, \n "

	<< "	}, \n "

	<< "	Geometry={ \n "

	<< "		Type    = \"Origin_DxDyDz\", \n "

	<< "		Origin  = {" << ToString(xmin_, ",") << "}, \n "

	<< "		DxDyDz  = {" << ToString(dx_, ",") << "}, \n "

	<< "		Min     = {" << ToString(xmin_, ",") << "}, \n "

	<< "		Max     = {" << ToString(xmax_, ",") << "}, \n "

	<< "		k       = {" << ToString(k_, ",") << "}, \n "

	<< "	}, \n "

	<< "	dt = " << GetDt() << ",\n"

	<< "\t" << constants << "\n"

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
        std::function<void(int, index_type, index_type, index_type)> const &fun, unsigned int flags) const
{

	index_type ib = ((flags & WITH_GHOSTS) <= 0) ? ghost_width_[0] : 0;
	index_type ie = ((flags & WITH_GHOSTS) <= 0) ? dims_[0] - ghost_width_[0] : dims_[0];

	index_type jb = ((flags & WITH_GHOSTS) <= 0) ? ghost_width_[1] : 0;
	index_type je = ((flags & WITH_GHOSTS) <= 0) ? dims_[1] - ghost_width_[1] : dims_[1];

	index_type kb = ((flags & WITH_GHOSTS) <= 0) ? ghost_width_[2] : 0;
	index_type ke = ((flags & WITH_GHOSTS) <= 0) ? dims_[2] - ghost_width_[2] : dims_[2];

	int mb = 0;
	int me = num_comps_per_cell_[IFORM];

	size_t len = ie - ib;
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

//**
//*  Boundary
//**/
//
//
//template<typename Fun> inline
//void ForEachBoundary(int iform, Fun const &f) const
//{
//	size_t num_comp = num_comps_per_cell_[iform];
//
//	for (size_t i = 0; i < dims_[0]; ++i)
//		for (size_t j = 0; j < dims_[1]; ++j)
//			for (size_t k = 0; k < dims_[2]; ++k)
//				for (int m = 0; m < num_comp; ++m)
//				{
//					if (i >= gw_[0] && i < dims_[0] - gw_[0] &&
//
//					j >= gw_[1] && j < dims_[1] - gw_[1] &&
//
//					k >= gw_[2] && k < dims_[2] - gw_[2]
//
//					)
//					{
//						continue;
//					}
//					else
//					{
//						f(
//								(i * strides_[0] + j * strides_[1]
//										+ k * strides_[2]) * num_comp + m);
//					}
//
//				}
//
//}
//
//void MakeCycleMap(int iform, std::map<index_type, index_type> &ma,
//		unsigned int flag = 7) const
//{
//	size_t num_comp = num_comps_per_cell_[iform];
//
//	nTuple<NUM_OF_DIMS, size_t> L =
//	{ dims_[0] - 2 * gw_[0], dims_[1] - 2 * gw_[1], dims_[2] - 2 * gw_[2] };
//
//	for (size_t i = 0; i < dims_[0]; ++i)
//		for (size_t j = 0; j < dims_[1]; ++j)
//			for (size_t k = 0; k < dims_[2]; ++k)
//			{
//
//				index_type s = i * strides_[0] + j * strides_[1]
//						+ k * strides_[2];
//				index_type t = s;
//
//				if (flag & 1)
//				{
//					if (i < gw_[0])
//					{
//						t += L[0] * strides_[0];
//					}
//					else if (i >= dims_[0] - gw_[0])
//					{
//						t -= L[0] * strides_[0];
//					}
//				}
//
//				if (flag & 2)
//				{
//					if (j < gw_[1])
//					{
//						t += L[1] * strides_[1];
//					}
//					else if (j >= dims_[1] - gw_[1])
//					{
//						t -= L[1] * strides_[1];
//					}
//				}
//
//				if (flag & 4)
//				{
//					if (k < gw_[2])
//					{
//						t += L[2] * strides_[2];
//					}
//					else if (k >= dims_[2] - gw_[2])
//					{
//						t -= L[2] * strides_[2];
//					}
//				}
//				if (s != t)
//				{
//					for (int m = 0; m < num_comp; ++m)
//					{
//						ma[s * num_comp + m] = t * num_comp + m;
//					}
//				}
//
//			}
//}
//
//template<int IFORM, typename T1, typename T2>
//void UpdateBoundary(std::map<index_type, index_type> const & m,
//		Field<Geometry<this_type, IFORM>, T1> & src,
//		Field<Geometry<this_type, IFORM>, T2> & dest) const
//{
//	for (auto & p : m)
//	{
//		dest[p.first] = src[p.second];
//	}
//
//}
//
//template<int IFORM, typename T1>
//void UpdateCyCleBoundary(Field<Geometry<this_type, IFORM>, T1> & f) const
//{
//	std::map<index_type, index_type> m;
//	MakeCycleMap(IFORM, m);
//	UpdateBoundary(m, f, f);
//}
/**
 *
 *
 // Interpolation ----------------------------------------------------------

 template<typename TExpr>
 inline typename Field<Geometry<this_type, 0>, TExpr>::Value //
 Gather(Field<Geometry<this_type, 0>, TExpr> const &f, RVec3 x) const
 {
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);

 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 return (f[s] * (1.0 - r[0]) + f[s + strides_[0]] * r[0]); //FIXME Only for 1-dim
 }

 template<typename TExpr>
 inline void //
 Scatter(Field<Geometry<this_type, 0>, TExpr> & f, RVec3 x,
 typename Field<Geometry<this_type, 0>, TExpr>::Value const v) const
 {
 typename Field<Geometry<this_type, 0>, TExpr>::Value res;
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);
 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 f.Add(s, v * (1.0 - r[0]));
 f.Add(s + strides_[0], v * r[0]); //FIXME Only for 1-dim

 }

 template<typename TExpr>
 inline nTuple<THREE, typename Field<Geometry<this_type, 1>, TExpr>::Value> //
 Gather(Field<Geometry<this_type, 1>, TExpr> const &f, RVec3 x) const
 {
 nTuple<THREE, typename Field<Geometry<this_type, 1>, TExpr>::Value> res;

 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx = r + 0.5;
 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 res[0] = (f[(s) * 3 + 0] * (0.5 - r[0])
 + f[(s - strides_[0]) * 3 + 0] * (0.5 + r[0]));
 res[1] = (f[(s) * 3 + 1] * (0.5 - r[1])
 + f[(s - strides_[1]) * 3 + 1] * (0.5 + r[1]));
 res[2] = (f[(s) * 3 + 2] * (0.5 - r[2])
 + f[(s - strides_[2]) * 3 + 2] * (0.5 + r[2]));
 return res;
 }
 template<typename TExpr>
 inline void //
 Scatter(Field<Geometry<this_type, 1>, TExpr> & f, RVec3 x,
 nTuple<THREE, typename Field<Geometry<this_type, 1>, TExpr>::Value> const &v) const
 {
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx = r + 0.5;
 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 f[(s) * 3 + 0] += v[0] * (0.5 - r[0]);
 f[(s - strides_[0]) * 3 + 0] += v[0] * (0.5 + r[0]);
 f[(s) * 3 + 1] += v[1] * (0.5 - r[1]);
 f[(s - strides_[1]) * 3 + 1] += v[1] * (0.5 + r[1]);
 f[(s) * 3 + 2] += v[2] * (0.5 - r[2]);
 f[(s - strides_[2]) * 3 + 2] += v[2] * (0.5 + r[2]);
 }

 template<typename TExpr>
 inline nTuple<THREE, typename Field<Geometry<this_type, 2>, TExpr>::Value> //
 Gather(Field<Geometry<this_type, 2>, TExpr> const &f, RVec3 x) const
 {
 nTuple<THREE, typename Field<Geometry<this_type, 2>, TExpr>::Value> res;

 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);

 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 res[0] = (f[(s) * 3 + 0] * (1.0 - r[0])
 + f[(s - strides_[0]) * 3 + 0] * (r[0]));
 res[1] = (f[(s) * 3 + 1] * (1.0 - r[1])
 + f[(s - strides_[1]) * 3 + 1] * (r[1]));
 res[2] = (f[(s) * 3 + 2] * (1.0 - r[2])
 + f[(s - strides_[2]) * 3 + 2] * (r[2]));
 return res;

 }

 template<typename TExpr>
 inline void //
 Scatter(Field<Geometry<this_type, 2>, TExpr> & f, RVec3 x,
 nTuple<THREE, typename Field<Geometry<this_type, 2>, TExpr>::Value> const &v) const
 {
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);

 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 f[(s) * 3 + 0] += v[0] * (1.0 - r[0]);
 f[(s - strides_[0]) * 3 + 0] += v[0] * (r[0]);
 f[(s) * 3 + 1] += v[1] * (1.0 - r[1]);
 f[(s - strides_[1]) * 3 + 1] += v[1] * (r[1]);
 f[(s) * 3 + 2] += v[2] * (1.0 - r[2]);
 f[(s - strides_[2]) * 3 + 2] += v[2] * (r[2]);

 }
 *
 *
 * */

#endif //UNIFORM_RECT_H_
