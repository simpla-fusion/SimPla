/*
 * field_io_cache.h
 *
 *  Created on: 2013年10月31日
 *      Author: salmon
 */

#ifndef FIELD_IO_CACHE_H_
#define FIELD_IO_CACHE_H_

#include <vector>

#include "../utilities/log.h"
#include "../utilities/type_utilites.h"
#include "../utilities/pretty_stream.h"
#include "field.h"
#include "cache.h"

namespace simpla
{

template<typename TGeometry, typename TValue>
class Field<TGeometry, Cache<const Field<TGeometry, TValue> > >
{

public:

	typedef TGeometry geometry_type;

	typedef Field<TGeometry, TValue> field_type;

	typedef Field<TGeometry, Cache<const Field<TGeometry, TValue> > > this_type;

	typedef typename geometry_type::mesh_type mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	static const int IForm = geometry_type::IForm;

	typedef typename field_type::value_type value_type;

	typedef typename field_type::field_value_type field_value_type;

	mesh_type const &mesh;

	field_value_type mean_;

private:
	field_type const & f_;

	index_type cell_idx_;

	int affect_region_;

	size_t num_of_points_;

	std::vector<index_type> points_;

	std::vector<value_type> cache_;

public:

	Field(this_type const& r)
			: mesh(r.mesh), f_(r.f_),

			cell_idx_(r.cell_idx_), affect_region_(r.affect_region_),

			num_of_points_(r.num_of_points_),

			points_(r.points_), cache_(r.cache_)

	{
	}

	Field(this_type && r)
			: mesh(r.mesh), f_(r.f_),

			cell_idx_(r.cell_idx_), affect_region_(r.affect_region_),

			num_of_points_(r.num_of_points_),

			points_(r.points_), cache_(r.cache_)

	{
	}

	Field(field_type const & f, index_type const &s, int affect_region = 2)
			: mesh(f.mesh), f_(f), cell_idx_(s), affect_region_(affect_region), num_of_points_(0)
	{
	}

	~Field()
	{
	}

	void RefreshCache(size_t s)
	{
		cell_idx_ = s;
		num_of_points_ = (mesh.GetAffectedPoints(Int2Type<IForm>(), cell_idx_, nullptr, affect_region_));
		if (num_of_points_ == 0)
		{
			WARNING << "Empty Cache!";
			return;
		}
		points_.resize(num_of_points_);
		cache_.resize(num_of_points_);

		mesh.GetAffectedPoints(Int2Type<IForm>(), cell_idx_, &points_[0], affect_region_);

		for (size_t i = 0; i < num_of_points_; ++i)
		{
			cache_[i] = f_[points_[i]];
		}

		UpdateMeanValue(Int2Type<IForm>());
	}

private:
	void UpdateMeanValue(Int2Type<0>)
	{
		mesh.template GetMeanValue<IForm>(&cache_[0], &mean_, affect_region_);
	}
	void UpdateMeanValue(Int2Type<3>)
	{
		mesh.template GetMeanValue<IForm>(&cache_[0], &mean_, affect_region_);
	}
	void UpdateMeanValue(Int2Type<1>)
	{
		mesh.template GetMeanValue<IForm>(&cache_[0], &mean_[0], affect_region_);
	}
	void UpdateMeanValue(Int2Type<2>)
	{
		mesh.template GetMeanValue<IForm>(&cache_[0], &mean_[0], affect_region_);
	}
public:
	inline field_value_type operator()(coordinates_type const &x) const
	{
		coordinates_type pcoords;

		index_type idx = mesh.SearchCell(cell_idx_, x, &(pcoords[0]));

		field_value_type res;

		if (idx == cell_idx_)
		{
			mesh.template Gather(Int2Type<IForm>(), &pcoords[0], &cache_[0], &res, affect_region_);
		}
		else //failsafe
		{
			res = f_(idx, &pcoords[0]);
		}

		return res;

	}

	inline field_value_type const & mean(coordinates_type const &) const
	{
		return mean_;
	}

}
;

template<typename TGeometry, typename TValue>
class Field<TGeometry, Cache<Field<TGeometry, TValue> *> >
{

public:

	typedef TGeometry geometry_type;

	typedef Field<TGeometry, TValue> field_type;

	typedef Field<geometry_type, Cache<field_type> > this_type;

	typedef typename geometry_type::mesh_type mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	static const int IForm = geometry_type::IForm;

	typedef typename field_type::value_type value_type;

	typedef typename field_type::field_value_type field_value_type;

	mesh_type const &mesh;
private:
	field_type * f_;

	index_type cell_idx_;

	int affect_region_;

	size_t num_of_points_;

	std::vector<index_type> points_;

	std::vector<value_type> cache_;

	bool is_fresh_;

public:

	Field(this_type && r)
			: mesh(r.mesh), f_(r.f_),

			cell_idx_(r.cell_idx_), affect_region_(r.affect_region_),

			num_of_points_(r.num_of_points_),

			points_(r.points_), cache_(r.cache_), is_fresh_(r.is_fresh_)
	{
	}
	Field(this_type const& r)
			: mesh(r.mesh), f_(r.f_),

			cell_idx_(r.cell_idx_), affect_region_(r.affect_region_),

			num_of_points_(r.num_of_points_),

			points_(r.points_), cache_(r.cache_), is_fresh_(r.is_fresh_)

	{
	}

	Field(field_type * f, int affect_region = 2)
			: mesh(f->mesh), f_(f), affect_region_(affect_region), num_of_points_(0), cell_idx_(0), is_fresh_(false)
	{
	}

	~Field()
	{
		FlushCache();
	}

	void FlushCache()
	{
		if (num_of_points_ > 0 && !is_fresh_)
		{
			f_->Collect(num_of_points_, &points_[0], &cache_[0]);
			is_fresh_ = true;
		}
	}

	void RefreshCache(size_t s)
	{
		cell_idx_ = s;

		num_of_points_ = (mesh.GetAffectedPoints(Int2Type<IForm>(), cell_idx_, nullptr, affect_region_));

		if (num_of_points_ == 0)
		{
			WARNING << "Empty Cache!";
			return;
		}
		points_.resize(num_of_points_);
		cache_.resize(num_of_points_);

		mesh.GetAffectedPoints(Int2Type<IForm>(), cell_idx_, &points_[0], affect_region_);

		value_type zero_value_;

		zero_value_ *= 0;

		std::fill(cache_.begin(), cache_.end(), zero_value_);

		is_fresh_ = false;
	}

	template<typename TV>
	inline void Collect(TV const &v, coordinates_type const &x)
	{
		coordinates_type pcoords;

		index_type idx = mesh.SearchCell(cell_idx_, x, &pcoords[0]);

		if (idx == cell_idx_)
		{
			field_value_type vv;
			vv = v;
			mesh.Scatter(Int2Type<IForm>(), &pcoords[0], vv, &cache_[0], affect_region_);
		}
		else //failsafe
		{
			f_->Collect(v, idx, &pcoords[0], affect_region_);
		}
	}

};

template<typename TGeometry, typename TValue>
struct Cache<const Field<TGeometry, TValue> >
{

	typedef Field<TGeometry, Cache<const Field<TGeometry, TValue> > > type;

	template<typename ... Args>
	Cache(Field<TGeometry, TValue> const & f, Args const & ... args)
			: f_(f, std::forward<Args const &>(args)...)
	{
		VERBOSE << "Field read cache applied!";
	}

	type & operator*()
	{
		return f_;
	}

	type const & operator*() const
	{
		return f_;
	}
private:
	type f_;
};

template<typename TGeometry, typename TValue>
struct Cache<Field<TGeometry, TValue>*>
{

	typedef Field<TGeometry, Cache<Field<TGeometry, TValue>*> > type;

	template<typename ... Args>
	Cache(Field<TGeometry, TValue>* f, Args const & ... args)
			: f_(f, std::forward<Args const &>(args)...)
	{
		VERBOSE << "Field write cache applied!";
	}

	type & operator*()
	{
		return f_;
	}
private:
	type f_;
};

template<typename TG, typename TF>
void RefreshCache(size_t s, Field<TG, Cache<TF>> & f)
{
	f.RefreshCache(s);
}

template<typename TG, typename TF>
void FlushCache(Field<TG, Cache<TF*>> & f)
{
	f.FlushCache();
}

}  // namespace simpla

#endif /* FIELD_IO_CACHE_H_ */
