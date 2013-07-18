/*
 * fetl/grid/dummy.h
 *
 *  Created on: 2012-1-15
 *      Author: salmon
 */

#ifndef _GRID_IS_DEFINED_
#define _GRID_IS_DEFINED_ "Dummy"

/** @file fetl/grid/dummy.h
 *
 *  Define the interface of grid.
 */
namespace simpla
{
class Grid
{
	Grid(const Grid &rhs);
	Grid & operator=(const Grid&);

public:
	typedef Grid ThisType;

	Grid();
	~Grid();

	template<typename TF> inline typename Array::Holder createField() const;

	template<typename TL, typename TR> void assign(Field<TL> & lhs,
			TR const& rhs) const;

	// Property -----------------------------------------------

	inline SizeType dim_min(int i) const;

	inline SizeType dim_max(int i) const;

	inline SizeType get_num_Of_comp(int iform) const;

	inline SizeType get_num_Of_ele(int iform) const;

	inline SizeType get_num_Of_grid_point() const;

	inline SizeType get_num_of_center_grid_point() const;

	inline SizeType get_num_of_cell(int iform = 0) const;

	inline IVec3 get_idx(SizeType s) const;

	inline SizeType get_cell_num(SizeType IX, SizeType IY, SizeType IZ) const;

	inline SizeType get_cell_num(const IVec3 & IX) const;

	inline SizeType get_cell_num(Vec3 x) const;

	inline Real get_cell_volumn() const;

	inline Real get_cell_d_volumn() const;

	template<typename TF> std::vector<SizeType> get_field_shape() const;

	template<typename TF> inline typename TF::Holder createField() const
	{
		return (typename TF::Holder(new TF(*this)));
	}
	// IO ----------------------------------------------------------

	template<typename TExpr> typename Field<TExpr>::EleValueType //
	gather(Field<TExpr> const &, RVec3 x);

	template<typename TExpr> void //
	scatter(Field<TExpr> const &, RVec3 x,
			typename Field<TExpr>::EleValueType const &);

};

namespace _detail
{

template<typename TExpr> inline typename Field<TExpr>::ValueType //
mapto_(Int2Type<IOneForm>, Int2Type<IZeroForm>, const Field<TExpr> & expr,
		SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr> inline typename Field<TExpr>::ValueType //
mapto_(Int2Type<ITwoForm>, Int2Type<IZeroForm>, const Field<TExpr> & expr,
		SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr> inline typename Field<TExpr>::ValueType //
mapto_(Int2Type<IVecZeroForm>, Int2Type<IZeroForm>, const Field<TExpr> & expr,
		SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr> inline typename Field<TExpr>::ValueType //
mapto_(Int2Type<IZeroForm>, Int2Type<IOneForm>, const Field<TExpr> & expr,
		SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr> inline typename Field<TExpr>::ValueType //
mapto_(Int2Type<IZeroForm>, Int2Type<ITwoForm>, const Field<TExpr> & expr,
		SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr> inline typename Field<TExpr>::ValueType //
mapto_(Int2Type<IZeroForm>, Int2Type<IVecZeroForm>, const Field<TExpr> & expr,
		SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr> inline typename Field<TExpr>::ValueType //
mapto_(Int2Type<IOneForm>, Int2Type<IVecZeroForm>, const Field<TExpr> & expr,
		SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr> inline typename Field<TExpr>::ValueType //
mapto_(Int2Type<ITwoForm>, Int2Type<IVecZeroForm>, const Field<TExpr> & expr,
		SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr> inline typename Field<TExpr>::ValueType //
mapto_(Int2Type<IVecZeroForm>, Int2Type<IOneForm>, const Field<TExpr> & expr,
		SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr> inline typename Field<TExpr>::ValueType //
mapto_(Int2Type<IVecZeroForm>, Int2Type<ITwoForm>, const Field<TExpr> & expr,
		SizeType s) const
{
#warning unimplement!
	return (1);
}

template<typename TL> inline //
typename Field<_FETL_Expr<opGrad, TL, Int2Type<INullForm> > >::ValueType //
grad_(Field<_FETL_Expr<opGrad, TL, Int2Type<INullForm> > > const & expr
		, SizeType s)

{
#warning unimplement!
	return (1);
}
template<typename TL> inline //
typename Field<_FETL_Expr<opDiverge, TL, Int2Type<INullForm> > >::ValueType //
diverge_(Field<_FETL_Expr<opDiverge, TL, Int2Type<INullForm> > > const & expr
		, SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr> inline typename Field<TExpr>::ValueType //
curl_(Int2Type<IOneForm>, Field<TExpr> const & expr, SizeType s)
{
#warning unimplement!
	return (1);
}
template<typename TExpr> inline typename Field<TExpr>::ValueType //
curl_(Int2Type<ITwoForm>, Field<TExpr> const & expr, SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr, int PD> inline typename Field<TExpr>::ValueType //
curlPd_(Int2Type<PD>, Int2Type<IOneForm>, const Field<TExpr> & expr, SizeType s)
{
#warning unimplement!
	return (1);
}

template<typename TExpr, int PD> inline typename Field<TExpr>::ValueType //
curlPd_(Int2Type<PD>, Int2Type<ITwoForm>, const Field<TExpr> & expr, SizeType s)
{
#warning unimplement!
	return (1);
}
} //namespace _detail
} //namespace simpla
#endif /* _GRID_IS_DEFINED_ */
