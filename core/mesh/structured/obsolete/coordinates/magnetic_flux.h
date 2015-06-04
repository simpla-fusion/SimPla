/**
 * @file  geometry_magnetic_flux.h
 *
 *  created on: 2014-3-13
 *      Author: salmon
 */

#ifndef GEOMETRY_MAGNETIC_FLUX_H_
#define GEOMETRY_MAGNETIC_FLUX_H_

#include <iostream>
#include <utility>

#include "../gtl/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/log.h"

namespace simpla
{
/**
 *  @ingroup Geometry
 *
 *  \brief  Magnetic flus coordinates ( R psi phi ) @todo !!!UNIMPLEMENTED !!!
 */
template<typename Topology>
struct MagneticFluxGeometry
{
	typedef Topology topology_type;

	typedef MagneticFluxGeometry<topology_type> this_type;

	static constexpr  unsigned int  NDIMS = topology_type::NDIMS;

	typedef typename topology_type::index_type index_type;
	typedef typename topology_type::coordinate_tuple coordinate_tuple;
	typedef typename topology_type::iterator iterator;

	typedef nTuple<NDIMS, Real> vector_type;
	typedef nTuple<NDIMS, Real> covector_type;

	topology_type const & topology;

	MagneticFluxGeometry(this_type const & rhs) = delete;

	MagneticFluxGeometry(topology_type const & t)
			: topology(t)
	{
		UNIMPLEMENTED;
	}
	template<typename TDict>
	MagneticFluxGeometry(topology_type const & t, TDict const & dict)
			: topology(t)
	{

	}

	~MagneticFluxGeometry()
	{

	}

	//***************************************************************************************************
	// Geometric properties
	// Metric
	//***************************************************************************************************

	void update()
	{

	}
	template<typename TDict>
	void load(TDict const & dict)
	{

	}

	std::ostream & save(std::ostream & os) const
	{
		return os;
	}

	coordinate_tuple xmin_ = { 0, 0, 0 };

	coordinate_tuple xmax_ = { 1, 1, 1 };

	coordinate_tuple inv_L = { 1.0, 1.0, 1.0 };

	coordinate_tuple scale_ = { 1.0, 1.0, 1.0 };

	coordinate_tuple inv_scale_ = { 1.0, 1.0, 1.0 };

	coordinate_tuple shift_ = { 0, 0, 0 };

	static constexpr nTuple<NDIMS, Real> normal_[NDIMS] = {

	1, 0, 0,

	0, 1, 0,

	0, 0, 1

	};

	/**
	 *
	 *                ^y
	 *               /
	 *        z     /
	 *        ^    /
	 *        |  110-------------111
	 *        |  /|              /|
	 *        | / |             / |
	 *        |/  |            /  |
	 *       100--|----------101  |
	 *        | m |           |   |
	 *        |  010----------|--011
	 *        |  /            |  /
	 *        | /             | /
	 *        |/              |/
	 *       000-------------001---> x
	 *
	 *
	 */
	Real dh_[NDIMS] = { 1, 1, 1 };

	Real inv_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	nTuple<NDIMS, Real> dx_;

	DenseContainer<index_type, coordinate_tuple> mesh_points_;

	DenseContainer<index_type, coordinate_tuple> volumes_[4];

	nTuple<NDIMS, Real> const & get_dx() const
	{
		return dx_;
	}

	template<unsigned int IN, typename T>
	inline void set_extents(nTuple<IN, T> const & pmin, nTuple<IN, T> const & pmax)
	{
		int n = IN < NDIMS ? IN : NDIMS;

		auto const & dims = topology.get_dimensions();

		for (int i = 0; i < n; ++i)
		{
			xmin_[i] = pmin[i];
			xmax_[i] = pmax[i];

			shift_[i] = pmin[i];

			scale_[i] = (pmax[i] > pmin[i]) ? ((static_cast<Real>(dims[i])) / (pmax[i] - pmin[i])) : 0;
			inv_scale_[i] = (pmax[i] - pmin[i]) / (static_cast<Real>(dims[i]));
		}

		/**
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^    /
		 *        |  110-------------111
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *       100--|----------101  |
		 *        | m |           |   |
		 *        |  010----------|--011
		 *        |  /            |  /
		 *        | /             | /
		 *        |/              |/
		 *       000-------------001---> x
		 *
		 *
		 */

		dh_[0] /* R   */= (dims[0] > 1) ? (xmax_[0] - xmin_[0]) / dims[0] : 1;
		dh_[1] /* Z   */= (dims[1] > 1) ? (xmax_[1] - xmin_[1]) / dims[1] : 1;
		dh_[2] /* Phi */= (dims[2] > 1) ? (xmax_[2] - xmin_[2]) / dims[2] : 1;

	}

	inline std::pair<coordinate_tuple, coordinate_tuple> get_extents() const
	{
		return std::move(std::make_pair(xmin_, xmax_));
	}

	nTuple<3, Real> const& Normal(iterator s) const
	{
		return normal_[topology.topology_type::_C(s)];
	}

	template<typename TV>
	TV const& Normal(iterator s, nTuple<3, TV> const & v) const
	{
		return v[topology.topology_type::_C(s)];
	}

	Real volume(iterator s) const
	{
		Real res = 1;

		return res;
	}
	Real inv_volume(iterator s) const
	{
		return 1.0 / volume(s);
	}

	coordinate_tuple coordinates_local_to_global(coordinate_tuple const &x) const
	{

		return coordinate_tuple( {

		x[0] * inv_scale_[0] + shift_[0],

		x[1] * inv_scale_[1] + shift_[1],

		x[2] * inv_scale_[2] + shift_[2]

		});
	}
	coordinate_tuple coordinates_global_to_local(coordinate_tuple const &x) const
	{
		return coordinate_tuple( {

		(x[0] - shift_[0]) * scale_[0],

		(x[1] - shift_[1]) * scale_[1],

		(x[2] - shift_[2]) * scale_[2]

		});
	}

	template<typename TV>
	nTuple<NDIMS, TV> const& push_forward(iterator s, nTuple<NDIMS, TV> const & v) const
	{
		return v;
	}
	template<typename TV>
	nTuple<NDIMS, TV> const& pull_back(iterator s, nTuple<NDIMS, TV> const & v) const
	{
		return v;
	}
}
;

}  // namespace simpla

#endif /* GEOMETRY_MAGNETIC_FLUX_H_ */
