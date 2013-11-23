/*
 * field_io_cache.h
 *
 *  Created on: 2013年10月31日
 *      Author: salmon
 */

#ifndef FIELD_IO_CACHE_H_
#define FIELD_IO_CACHE_H_

#include <vector>
#include <algorithm>
#include "field.h"
#include "primitives.h"
#include "proxycache.h"

namespace simpla
{

template<typename TGeometry, typename TValue>
class Field<TGeometry, ProxyCache<const Field<TGeometry, TValue> > >
{

public:

	typedef TGeometry geometry_type;

	typedef const Field<TGeometry, TValue> field_type;

	typedef Field<geometry_type, ProxyCache<field_type> > this_type;

	typedef typename geometry_type::mesh_type mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	static const int IForm = geometry_type::IForm;

	typedef typename field_type::value_type value_type;

	typedef typename field_type::field_value_type field_value_type;

	mesh_type const &mesh;

	Field(this_type && r) :
			mesh(r.mesh), cell_idx_(r.cell_idx_), affect_region_(
					r.affect_region_), f_(r.f_), zero_value_(r.zero_value_), points_(
					r.points_), cache_(r.cache_)
	{

	}

	Field(field_type const & f, index_type const &s, int affect_region = 1) :
			mesh(f.mesh), cell_idx_(s), affect_region_(affect_region), f_(f)
	{
		mesh.GetAffectedPoints(Int2Type<IForm>(), cell_idx_, points_,
				affect_region_);

		for (auto const &p : points_)
		{
			cache_.push_back(f_[p]);
		}
		zero_value_ *= 0;
		zero_field_value_ *= 0;
	}

	~Field()
	{
	}

	inline field_value_type operator()(coordinates_type const &x) const
	{
		return std::move(Gather(x));
	}

	inline field_value_type Gather(coordinates_type const &x) const
	{
		coordinates_type pcoords;

		std::vector<typename geometry_type::weight_type> weights(
				points_.size());

		index_type idx = mesh.SearchCell(cell_idx_, x, &(pcoords));

		if (idx == cell_idx_)
		{
			mesh.CalcuateWeights(Int2Type<IForm>(), pcoords, weights,
					affect_region_);

			return std::move(
					std::inner_product(weights.begin(), weights.end(),
							cache_.begin(), zero_field_value_));
		}
		else
		{
			return std::move(f_.Gather(idx, pcoords));
		}

	}

private:
	field_type & f_;

	index_type cell_idx_;

	int affect_region_;

	value_type zero_value_;

	field_value_type zero_field_value_;

	std::vector<index_type> points_;

	std::vector<value_type> cache_;
};

template<typename TGeometry, typename TValue>
class Field<TGeometry, ProxyCache<Field<TGeometry, TValue> > >
{

public:

	typedef TGeometry geometry_type;

	typedef Field<TGeometry, TValue> field_type;

	typedef Field<geometry_type, ProxyCache<field_type> > this_type;

	typedef typename geometry_type::mesh_type mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	static const int IForm = geometry_type::IForm;

	typedef typename field_type::value_type value_type;

	typedef typename field_type::field_value_type field_value_type;

	mesh_type const &mesh;

	Field(this_type && r) :
			mesh(r.mesh), cell_idx_(r.cell_idx_), affect_region_(
					r.affect_region_), f_(r.f_), zero_value_(r.zero_value_), points_(
					r.points_), cache_(r.cache_)
	{

//		mesh.GetAffectedPoints(Int2Type<IForm>(), cell_idx_, points_,
//				affect_region_);
//
//		for (auto const &p : points_)
//		{
//			cache_.push_back(zero_value_);
//		}
	}

	Field(field_type & f, index_type const &s, int affect_region = 1) :
			mesh(f.mesh), cell_idx_(s), affect_region_(affect_region), f_(f)
	{

		mesh.GetAffectedPoints(Int2Type<IForm>(), cell_idx_, points_,
				affect_region_);
		zero_value_ *= 0;
		for (auto const &p : points_)
		{
			cache_.push_back(zero_value_);
		}

	}

	~Field()
	{
		f_.Scatter(points_, cache_);
	}

//	template<typename TV>
	inline void Scatter(field_value_type const &v, coordinates_type const &x)
	{
		coordinates_type pcoords;

		std::vector<typename geometry_type::weight_type> weights;

		index_type idx = mesh.SearchCell(cell_idx_, x, &pcoords);

		if (idx == cell_idx_)
		{
			mesh.CalcuateWeights(Int2Type<IForm>(), pcoords, weights,
					affect_region_);

			auto it1 = cache_.begin();
			auto it2 = weights.begin();
			auto it2_end = weights.end();

			for (; it1 != cache_.end() && it2 != it2_end; ++it1, ++it2)
			{
				// FIXME: this incorrect for vector field interpolation
				*it1 += Dot(v, (*it2));
			}
		}
		else
		{
			f_.Scatter(v, idx, pcoords, affect_region_);
		}
	}

private:
	field_type & f_;

	index_type cell_idx_;

	int affect_region_;

	value_type zero_value_;

	std::vector<index_type> points_;

	std::vector<value_type> cache_;
};

template<typename TGeometry, typename TValue>
struct ProxyCache<const Field<TGeometry, TValue> >
{
	typedef const Field<TGeometry, TValue> src_type;

	typedef Field<TGeometry, ProxyCache<src_type> > type;

	template<typename TI>
	static inline type Eval(src_type & f, TI const &hint_idx)
	{
		return std::move(type(f, hint_idx));
	}

};

template<typename TGeometry, typename TValue>
struct ProxyCache<Field<TGeometry, TValue> >
{
	typedef Field<TGeometry, TValue> src_type;

	typedef Field<TGeometry, ProxyCache<src_type> > type;

	template<typename TI>
	static inline type Eval(src_type & f, TI const & hint_idx)
	{
		return std::move(type(f, hint_idx));
	}
};

}  // namespace simpla

#endif /* FIELD_IO_CACHE_H_ */
