/*
 * geometry_magnetic_flux.h
 *
 *  created on: 2014-3-13
 *      Author: salmon
 */

#ifndef GEOMETRY_MAGNETIC_FLUX_H_
#define GEOMETRY_MAGNETIC_FLUX_H_

#include <iostream>
#include <utility>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/log.h"

namespace simpla
{
/**
 *  \ingroup Geometry
 *
 *  \brief  Magnetic flus coordinates ( R psi phi ) @todo !!!UNIMPLEMENT !!!
 */
template<typename Topology>
struct MagneticFluxGeometry
{
	typedef Topology topology_type;

	typedef MagneticFluxGeometry<topology_type> this_type;

	static constexpr  unsigned int  NDIMS = topology_type::NDIMS;

	typedef typename topology_type::compact_index_type compact_index_type;
	typedef typename topology_type::coordinates_type coordinates_type;
	typedef typename topology_type::iterator iterator;

	typedef nTuple<NDIMS, Real> vector_type;
	typedef nTuple<NDIMS, Real> covector_type;

	topology_type const & topology;

	MagneticFluxGeometry(this_type const & rhs) = delete;

	MagneticFluxGeometry(topology_type const & t)
			: topology(t)
	{
		UNIMPLEMENT;
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

	void Update()
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

	coordinates_type xmin_ = { 0, 0, 0 };

	coordinates_type xmax_ = { 1, 1, 1 };

	coordinates_type inv_L = { 1.0, 1.0, 1.0 };

	coordinates_type scale_ = { 1.0, 1.0, 1.0 };

	coordinates_type inv_scale_ = { 1.0, 1.0, 1.0 };

	coordinates_type shift_ = { 0, 0, 0 };

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

	DenseContainer<compact_index_type, coordinates_type> mesh_points_;

	DenseContainer<compact_index_type, coordinates_type> volumes_[4];

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

	inline std::pair<coordinates_type, coordinates_type> get_extents() const
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

	Real Volume(iterator s) const
	{
		Real res = 1;

		return res;
	}
	Real InvVolume(iterator s) const
	{
		return 1.0 / Volume(s);
	}

	coordinates_type CoordinatesLocalToGlobal(coordinates_type const &x) const
	{

		return coordinates_type( {

		x[0] * inv_scale_[0] + shift_[0],

		x[1] * inv_scale_[1] + shift_[1],

		x[2] * inv_scale_[2] + shift_[2]

		});
	}
	coordinates_type CoordinatesGlobalToLocal(coordinates_type const &x) const
	{
		return coordinates_type( {

		(x[0] - shift_[0]) * scale_[0],

		(x[1] - shift_[1]) * scale_[1],

		(x[2] - shift_[2]) * scale_[2]

		});
	}

	template<typename TV>
	nTuple<NDIMS, TV> const& PushForward(iterator s, nTuple<NDIMS, TV> const & v) const
	{
		return v;
	}
	template<typename TV>
	nTuple<NDIMS, TV> const& PullBack(iterator s, nTuple<NDIMS, TV> const & v) const
	{
		return v;
	}
}
;

}  // namespace simpla

#endif /* GEOMETRY_MAGNETIC_FLUX_H_ */
