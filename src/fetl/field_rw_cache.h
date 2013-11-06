/*
 * field_io_cache.h
 *
 *  Created on: 2013年10月31日
 *      Author: salmon
 */

#ifndef FIELD_IO_CACHE_H_
#define FIELD_IO_CACHE_H_

#include <fetl/field.h>
#include <fetl/geometry.h>
#include <fetl/primitives.h>
#include <type_traits>
#include <vector>

namespace simpla
{

template<typename TF> class RWCache;
template<typename TF> class ReadCache;
template<typename TF> class WriteCache;

template<typename TGeometry, typename TValue>
class RWCache<Field<TGeometry, TValue> >
{
public:
	typedef TGeometry geometry_type;

	typedef Field<TGeometry, TValue> field_type;

	typedef typename field_type::mesh_type mesh_type;

	static const int IForm = field_type::IForm;

	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinate_type coordinate_type;
	typedef typename field_type::value_type value_type;
	typedef typename field_type::field_value_type field_value_type;

	typedef RWCache<Field<TGeometry, TValue> > this_type;

	RWCache(this_type const & r) :
			cell_idx_(r.cell_idx_), mesh_(r.mesh_), affect_region_(
					r.affect_region_), points_(r.points_)
	{
		zero_value_ *= 0;
	}

	RWCache(mesh_type const & m, index_type const &s, int affect_region = 1) :
			cell_idx_(s), mesh_(m), affect_region_(affect_region)
	{
		mesh_.GetEffectedPoints(Int2Type<IForm>(), s, points_, affect_region_);
		zero_value_ *= 0;
	}

	~RWCache()
	{
	}

private:

	index_type cell_idx_;

	mesh_type const & mesh_;

	int affect_region_;

	field_value_type zero_value_;

	std::vector<index_type> points_;

	std::vector<value_type> cache_;

	std::vector<typename geometry_type::weight_type> weights_;

	coordinate_type pcoords_;
};

template<typename TGeometry, typename TValue>
class ReadCache<Field<TGeometry, TValue> > : public RWCache<
		Field<TGeometry, TValue> >
{

public:

	typedef TGeometry geometry_type;
	typedef Field<geometry_type, TValue> field_type;
	typedef typename field_type::mesh_type mesh_type;
	static const int IForm = field_type::IForm;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinate_type coordinate_type;
	typedef typename field_type::value_type value_type;
	typedef typename field_type::field_value_type field_value_type;

	typedef RWCache<field_type> base_type;
	typedef ReadCache<field_type> this_type;

	ReadCache(this_type const & r) :
			base_type(r), f_(r.f_)
	{
	}
	ReadCache(field_type const & f, index_type const &s, int affect_region = 1) :
			base_type(f.mesh, s, affect_region), f_(f)
	{
		for (auto & p : base_type::points_)
		{
			base_type::cache_.push_bach(f_[p]);
		}

	}

	~ReadCache()
	{
	}
	inline field_value_type operator()(coordinate_type const &x)
	{
		return std::move(Gather(x));
	}
	inline field_value_type Gather(coordinate_type const &x)
	{
		index_type idx = base_type::mesh_.SearchCell(base_type::cell_idx_, x,
				base_type::pcoords_);

		if (idx == base_type::cell_idx_)
		{
			base_type::mesh_.CalcuateWeight(Int2Type<IForm>(),
					base_type::pcoords_, base_type::weight_,
					base_type::affect_region_);

			return std::move(
					std::inner_product(base_type::weight_.begin(),
							base_type::weight_.end(), base_type::cache_.begin(),
							base_type::zero_value_));
		}
		else
		{
			return std::move(f_.Gather(idx, base_type::pcoords_));
		}

	}

private:
	field_type const & f_;
};

template<typename TGeometry, typename TValue>
class WriteCache<Field<TGeometry, TValue> > : public RWCache<
		Field<TGeometry, TValue> >
{

public:
	typedef TGeometry geometry_type;
	typedef Field<geometry_type, TValue> field_type;
	typedef typename field_type::mesh_type mesh_type;
	static const int IForm = field_type::IForm;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinate_type coordinate_type;
	typedef typename field_type::value_type value_type;
	typedef typename field_type::field_value_type field_value_type;

	typedef RWCache<field_type> base_type;
	typedef WriteCache<field_type> this_type;

	WriteCache(this_type const & r) :
			base_type(r), f_(r.f_)
	{
	}

	WriteCache(field_type & f, index_type const &s, int affect_region = 1) :
			base_type(f.mesh, s, affect_region), f_(f)
	{
		std::fill(base_type::cache_.begin(), base_type::cache_.end(),
				base_type::zero_value_);
	}
	~WriteCache()
	{
		f_.Scatter(base_type::points_, base_type::cache_);
	}

	inline void Scatter(field_value_type const &v, coordinate_type const &x)
	{
		index_type idx = base_type::mesh_.SearchCell(base_type::cell_idx_, x,
				base_type::pcoords_);

		if (idx == base_type::cell_idx_)
		{
			base_type::mesh_.CalcuateWeight(Int2Type<IForm>(),
					base_type::pcoords_, base_type::weights_,
					base_type::affect_region_);

			for (auto it1 = base_type::cache_.begin(), it2 =
					base_type::weights_.begin();
					it1 != base_type::cache_.end()
							&& it2 != base_type::weights_.end(); ++it1, ++it2)
			{
				// FIXME: this incorrect for vector field interpolation
				*it1 += Dot(v, *it2);
			}
		}
		else
		{
			f_.Scatter(v, idx, base_type::pcoords_, base_type::affect_region_);
		}
	}

private:

	field_type & f_;

};

template<typename IDX>
class MakeCache
{
	typedef IDX index_type;
	index_type s_;
public:
	MakeCache()
	{

	}
	MakeCache(index_type const &s) :
			s_(s)
	{
	}

	~MakeCache()
	{
	}

	inline void SetIndex(index_type const &s)
	{
		s_ = s;
	}
	inline index_type GetIndex()
	{
		return s_;
	}

	template<typename TF> inline TF & Eval(TF f) const
	{
		return std::forward<TF>(f);
	}

	template<typename TGeometry, typename TValue> inline typename std::enable_if<
			!std::is_const<Field<TGeometry, TValue> >::value,
			WriteCache<Field<TGeometry, TValue> > >::type Eval(
			Field<TGeometry, TValue> & f)
	{
		return std::move(WriteCache<Field<TGeometry, TValue> >(f, s_));
	}

	template<typename TGeometry, typename TValue> inline typename std::enable_if<
			std::is_const<Field<TGeometry, TValue> >::value,
			ReadCache<Field<TGeometry, TValue> > >::type Eval(
			Field<TGeometry, TValue> const & f)
	{
		return std::move(ReadCache<Field<TGeometry, TValue> >(f, s_));
	}
};
}  // namespace simpla

#endif /* FIELD_IO_CACHE_H_ */
