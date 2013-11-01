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

	template<typename Element> using Container = std::vector<Element>;

	template<int IF> using weight_type =
	typename GeometryTraits<Geometry<this_type, IF> >::weight_type;

	typedef size_t index_type;

	typedef nTuple<3, Real> coordinates_type;

	this_type & operator=(const this_type&) = delete;

	Real dt_ = 0.0;

	MeshTraits();

	~MeshTraits() = default;

	inline bool operator==(this_type const & r) const;

	template<typename TCONFIG> void Config(TCONFIG const & vm);

	void Update();

	std::string Summary() const;

	template<int IFORM, typename T1>
	void Print(Field<Geometry<this_type, IFORM>, T1> const & f) const;

	template<typename E> inline Container<E>
	MakeContainer(int iform, E const & d = E()) const;

	void MakeCycleMap(int iform, std::map<index_type, index_type> &ma,
			unsigned int flag = 7) const;

	template<typename Fun> inline
	void ForAll(int iform, Fun const &f) const;

	template<typename Fun> inline
	void ForEach(int iform, Fun const &f) const;

	template<typename Fun> inline
	void ForEachBoundary(int iform, Fun const &f) const;

	template<int IFORM, typename T1, typename T2>
	void UpdateBoundary(std::map<index_type, index_type> const & m,
			Field<Geometry<this_type, IFORM>, T1> & src,
			Field<Geometry<this_type, IFORM>, T2> & dest) const;

	template<int IFORM, typename T1>
	void UpdateCyCleBoundary(Field<Geometry<this_type, IFORM>, T1> & f) const;

	// General Property -----------------------------------------------

	Real GetDt() const;

	void SetDt(Real dt = 0.0);

	inline size_t GetNumOfGridPoints(int iform) const;

	template<int IF> inline Real GetCellVolume(Int2Type<IF>) const;

	template<int IF> inline Real GetDCellVolume(Int2Type<IF>) const;

	// Searching Cell
	inline index_type SearchCell(coordinates_type const &x,
			coordinates_type *pcoords = nullptr) const;

	inline index_type SearchCell(index_type hint, coordinates_type const &x,
			coordinates_type *pcoords = nullptr) const;

	inline coordinates_type GetGlobalCoordinate(index_type s,
			coordinates_type const &r) const;

	template<int IF>
	inline void GetAffectedPoints(Int2Type<IF>, index_type const & idx,
			std::vector<index_type> & points, int affect_region = 1) const;

	template<int IF>
	inline void CalcuateWeights(Int2Type<IF>, coordinates_type const &pcoords,
			std::vector<weight_type<IF>> & weight, int affect_region = 1) const;

};

/**

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

 * */

} //namespace simpla

#endif /* MESH_H_ */
