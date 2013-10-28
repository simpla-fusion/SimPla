/*
 * mesh.h
 *
 *  Created on: 2012-10-26
 *      Author: salmon
 */

#ifndef MESH_H_
#define MESH_H_

#include <fetl/ntuple.h>
#include <fetl/primitives.h>
#include <cstddef>
#include <vector>

namespace simpla
{

template<typename, int> class Geometry;
template<typename, typename > class Field;

/**
 * @brief  Uniform interface to all mesh.
 * @ingroup Field Expression
 *
 */
template<typename _Mesh>
struct MeshTraits
{
	typedef MeshTraits<_Mesh> this_type;

	static const int NUM_OF_DIMS = 3;

	template<typename Element> using Container=std::vector<Element>;

	typedef size_t index_type;

	typedef size_t size_type;

	typedef nTuple<NUM_OF_DIMS, Real> coordinates_type;

	void Init();

	std::string Summary() const;

	size_type get_number_elements(int iform);

	coordinates_type get_coordinates(int iform, index_type const &);

	void get_coordinates(int iform, int num, index_type const idxs[],
			coordinates_type *x);

	size_t get_cell_number_vertices(int iform, index_type const &idx);

	void get_cell_vertices(int iform, index_type const &idx, index_type idxs[]);

	index_type get_cell_index(coordinates_type const &);

	Real get_cell_volumn(index_type const &idx) const;

	Real get_cell_d_volumn(index_type const &idx) const;

	//  ------------------------------------

	void CoordTransLocal2Global(int form, index_type id,
			coordinates_type const & r, coordinates_type *x);

	index_type CoordTransGlobal2Local(int form, coordinates_type const & x,
			index_type vertices[], Real weight[]) const
	{

	}

	// Interpolation ----------------------------------------------------------

	template<typename TExpr>
	inline typename Field<Geometry<Grid, 0>, TExpr>::Value //
	Gather(Field<Geometry<Grid, 0>, TExpr> const &f, coordinates_type) const;

	template<typename TExpr> inline void
	Scatter(Field<Geometry<Grid, 0>, TExpr> & f, coordinates_type,
			typename Field<Geometry<Grid, 0>, TExpr>::Value const v) const;

	template<typename TExpr> inline nTuple<THREE,
			typename Field<Geometry<Grid, 1>, TExpr>::Value>
	Gather(Field<Geometry<Grid, 1>, TExpr> const &f, coordinates_type) const;

	template<typename TExpr> inline void
	Scatter(Field<Geometry<Grid, 1>, TExpr> & f, coordinates_type,
			nTuple<THREE, typename Field<Geometry<Grid, 1>, TExpr>::Value> const &v) const;

	template<typename TExpr> inline nTuple<THREE,
			typename Field<Geometry<Grid, 2>, TExpr>::Value>
	Gather(Field<Geometry<Grid, 2>, TExpr> const &f, coordinates_type) const;

	template<typename TExpr> inline void
	Scatter(Field<Geometry<Grid, 2>, TExpr> & f, coordinates_type,
			nTuple<THREE, typename Field<Geometry<Grid, 2>, TExpr>::Value> const &v) const;

	// Mapto ----------------------------------------------------------
	/**
	 *    mapto(Int2Type<0> ,   //target topology position
	 *     Field<Grid,1 , TExpr> const & vl,  //field
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
	//
	//	template<int IF, typename TR> inline auto //
	//	mapto(Int2Type<IF>, TR const &l, size_t s) const DECL_RET_TYPE((l[s]))
	template<int IF> inline double mapto(Int2Type<IF>, double l,
			size_t s) const;

	template<int IF> inline std::complex<double>  //
	mapto(Int2Type<IF>, std::complex<double> l, size_t s) const;

	template<int IF, int N, typename TR> inline nTuple<N, TR>                //
	mapto(Int2Type<IF>, nTuple<N, TR> l, size_t s) const;

	template<int IF, typename TL> inline auto //
	mapto(Int2Type<IF>, Field<Geometry<this_type, IF>, TL> const &l,
			size_t s) const;

	template<typename TL> inline auto //
	mapto(Int2Type<1>, Field<Geometry<this_type, 0>, TL> const &l,
			size_t s) const;

	template<typename TL> inline auto //
	mapto(Int2Type<2>, Field<Geometry<this_type, 0>, TL> const &l,
			size_t s) const;
	template<typename TL> inline auto //
	mapto(Int2Type<3>, Field<Geometry<this_type, 0>, TL> const &l,
			size_t s) const;

	//-----------------------------------------
	// Vector Arithmetic
	//-----------------------------------------

	template<int N, typename TL> inline auto
	ExtriorDerivative(Field<Geometry<this_type, N>, TL> const & f,
			index_type s) const;

	template<typename TExpr> inline auto
	Grad(Field<Geometry<this_type, 0>, TExpr> const & f, index_type) const;

	template<typename TExpr> inline auto
	Diverge(Field<Geometry<this_type, 1>, TExpr> const & f, index_type) const;

	template<typename TL> inline auto
	Curl(Field<Geometry<this_type, 1>, TL> const & f, index_type) const;

	template<typename TL> inline auto
	Curl(Field<Geometry<this_type, 2>, TL> const & f, index_type) const;

	template<typename TExpr> inline auto
	CurlPD(Int2Type<1>, TExpr const & expr, index_type) const;

	template<typename TExpr> inline auto
	CurlPD(Int2Type<2>, TExpr const & expr, index_type) const;

	template<int IL, int IR, typename TL, typename TR> inline auto
	Wedge(Field<Geometry<this_type, IL>, TL> const &l,
			Field<Geometry<this_type, IR>, TR> const &r, index_type) const;

	template<int N, typename TL> inline auto
	HodgeStar(Field<Geometry<this_type, N>, TL> const & f, index_type) const;

	template<int N, typename TL> inline auto
	Negate(Field<Geometry<this_type, N>, TL> const & f, index_type) const;

	template<int IL, typename TL, typename TR> inline auto
	Plus(Field<Geometry<this_type, IL>, TL> const &l,
			Field<Geometry<this_type, IL>, TR> const &r, index_type) const;

	template<int IL, typename TL, typename TR> inline auto
	Minus(Field<Geometry<this_type, IL>, TL> const &l,
			Field<Geometry<this_type, IL>, TR> const &r, index_type) const;

	template<int IL, int IR, typename TL, typename TR> inline auto
	Multiplies(Field<Geometry<this_type, IL>, TL> const &l,
			Field<Geometry<this_type, IR>, TR> const &r, index_type) const;

	template<int IL, typename TL, typename TR> inline auto
	Multiplies(Field<Geometry<this_type, IL>, TL> const &l, TR r,
			index_type) const;

	template<int IR, typename TL, typename TR> inline auto
	Multiplies(TL l, Field<Geometry<this_type, IR>, TR> const & r,
			index_type) const;

	template<int IL, typename TL, typename TR> inline auto
	Divides(Field<Geometry<this_type, IL>, TL> const &l, TR const &r,
			index_type) const;

};

} //namespace simpla

#endif /* MESH_H_ */
