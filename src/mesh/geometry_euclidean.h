/*
 * geometry_euclidean.h
 *
 *  Created on: 2014年3月13日
 *      Author: salmon
 */

#ifndef GEOMETRY_EUCLIDEAN_H_
#define GEOMETRY_EUCLIDEAN_H_

#include <iostream>
#include <utility>

#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"

namespace simpla
{
template<typename TTopology>
struct EuclideanGeometry
{
	typedef TTopology topology_type;

	typedef EuclideanGeometry<topology_type> this_type;

	static constexpr int NDIMS = topology_type::NDIMS;

	typedef typename topology_type::coordinates_type coordinates_type;
	typedef typename topology_type::index_type index_type;

	typedef nTuple<NDIMS, Real> vector_type;
	typedef nTuple<NDIMS, Real> covector_type;

	topology_type const & topology;

	EuclideanGeometry(this_type const & rhs) = delete;

	EuclideanGeometry(topology_type const & t)
			: topology(t)
	{

	}
	template<typename TDict>
	EuclideanGeometry(topology_type const & t, TDict const & dict)
			: topology(t)
	{
	}

	~EuclideanGeometry()
	{
	}

	//***************************************************************************************************
	// Geometric properties
	// Metric
	//***************************************************************************************************

	coordinates_type xmin_ = { 0, 0, 0 };

	coordinates_type xmax_ = { 1, 1, 1 };

	coordinates_type scale_ = { 1.0, 1.0, 1.0 };

	coordinates_type inv_scale_ = { 1.0, 1.0, 1.0 };

	coordinates_type shift_ = { 0, 0, 0 };

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

	Real volume_[8] = { 1, // 000
	        1, //001
	        1, //010
	        1, //011
	        1, //100
	        1, //101
	        1, //110
	        1  //111
	        };
	Real inv_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	Real dual_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	Real inv_dual_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	template<typename TDict, typename ...Others>
	void Load(TDict const & dict, Others const &...)
	{
		if (dict["Min"] && dict["Max"])
		{
			LOGGER << "Load EuclideanGeometry ";
			SetExtent(dict["Min"].template as<nTuple<3, Real>>(), dict["Max"].template as<nTuple<3, Real>>());
			Update();
		}
	}

	std::string Dump(std::string const &path) const
	{
		std::stringstream os;

		os << "\tMin = " << xmin_ << " , " << "Max  = " << xmax_;

		return os.str();
	}

	template<typename T>
	inline void SetExtent(nTuple<NDIMS, T> const & pmin, nTuple<NDIMS, T> const & pmax)
	{
		for (int i = 0; i < NDIMS; ++i)
		{
			xmin_[i] = pmin[i];
			xmax_[i] = pmax[i];
		}

	}

	inline std::pair<coordinates_type, coordinates_type> GetExtent() const
	{
		return std::move(std::make_pair(xmin_, xmax_));
	}

	inline coordinates_type GetDx() const
	{
		auto dims = topology.GetDimensions();

		return std::move(coordinates_type( {

		(xmax_[0] - xmin_[0]) / static_cast<Real>(dims[0]),

		(xmax_[1] - xmin_[1]) / static_cast<Real>(dims[1]),

		(xmax_[2] - xmin_[2]) / static_cast<Real>(dims[2])

		}));
	}

	void Update()
	{
		auto const & dims = topology.GetDimensions();

		for (int i = 0; i < NDIMS; ++i)
		{
			shift_[i] = xmin_[i];
			scale_[i] = (xmax_[i] > xmin_[i]) ? ((static_cast<Real>(dims[i])) / (xmax_[i] - xmin_[i])) : 0;
			inv_scale_[i] = (xmax_[i] - xmin_[i]) / (static_cast<Real>(dims[i]));
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

		volume_[0] = 1;
		volume_[1] /* 001 */= (dims[0] > 1) ? (xmax_[0] - xmin_[0]) / static_cast<Real>(dims[0]) : 1;
		volume_[2] /* 010 */= (dims[1] > 1) ? (xmax_[1] - xmin_[1]) / static_cast<Real>(dims[1]) : 1;
		volume_[4] /* 100 */= (dims[2] > 1) ? (xmax_[2] - xmin_[2]) / static_cast<Real>(dims[2]) : 1;

		volume_[3] /* 011 */= volume_[1] * volume_[2];
		volume_[5] /* 101 */= volume_[4] * volume_[1];
		volume_[6] /* 110 */= volume_[2] * volume_[4];

		volume_[7] /* 111 */= volume_[1] * volume_[2] * volume_[4];

		dual_volume_[7] = 1;
		dual_volume_[6] /* 001 */= (dims[0] > 1) ? (xmax_[0] - xmin_[0]) / static_cast<Real>(dims[0]) : 1;
		dual_volume_[5] /* 010 */= (dims[1] > 1) ? (xmax_[1] - xmin_[1]) / static_cast<Real>(dims[1]) : 1;
		dual_volume_[3] /* 100 */= (dims[2] > 1) ? (xmax_[2] - xmin_[2]) / static_cast<Real>(dims[2]) : 1;

		dual_volume_[4] /* 011 */= dual_volume_[6] * dual_volume_[5];
		dual_volume_[2] /* 101 */= dual_volume_[3] * dual_volume_[6];
		dual_volume_[1] /* 110 */= dual_volume_[5] * dual_volume_[3];

		dual_volume_[0] /* 111 */= dual_volume_[6] * dual_volume_[5] * dual_volume_[3];

		for (int i = 0; i < 8; ++i)
		{
			inv_volume_[i] = 1.0 / volume_[i];
			inv_dual_volume_[i] = 1.0 / dual_volume_[i];
		}

	}

	inline coordinates_type GetCoordinates(coordinates_type const &x) const
	{
		return coordinates_type( {

		xmin_[0] + (xmax_[0] - xmin_[0]) * x[0],

		xmin_[1] + (xmax_[1] - xmin_[1]) * x[1],

		xmin_[2] + (xmax_[2] - xmin_[2]) * x[2]

		});
	}

	template<typename TV>
	TV const& Normal(index_type s, TV const & v) const
	{
		return v;
	}

	template<typename TV>
	TV const& Normal(index_type s, nTuple<3, TV> const & v) const
	{
		return v[topology.topology_type::_C(s)];
	}

	template<typename TV>
	TV Sample(Int2Type<VERTEX>, index_type s, TV const &v) const
	{
		return v * Volume(s);
	}

	template<typename TV>
	TV Sample(Int2Type<VOLUME>, index_type s, TV const &v) const
	{
		return v * Volume(s);
	}

	template<typename TV>
	TV Sample(Int2Type<EDGE>, index_type s, nTuple<3, TV> const &v) const
	{
		return v[topology.topology_type::_C(s)] * Volume(s);
	}

	template<typename TV>
	TV Sample(Int2Type<FACE>, index_type s, nTuple<3, TV> const &v) const
	{
		return v[topology.topology_type::_C(s)] * Volume(s);
	}

	Real Volume(index_type s) const
	{
		return topology.Volume(s) * volume_[topology._N(s)];
	}
	Real InvVolume(index_type s) const
	{
		return topology.InvVolume(s) * inv_volume_[topology._N(s)];
	}

	Real DualVolume(index_type s) const
	{
		return topology.DualVolume(s) * dual_volume_[topology._N(s)];
	}
	Real InvDualVolume(index_type s) const
	{
		return topology.InvDualVolume(s) * inv_dual_volume_[topology._N(s)];
	}

	//***************************************************************************************************
	// Cell-wise operation
	//***************************************************************************************************

	template<int IFORM, typename TV>
	TV Sample(Int2Type<IFORM>, index_type s, TV const & v) const
	{
		return v * Volume(s);
	}

	template<int IFORM, typename TV>
	typename std::enable_if<(IFORM == EDGE || IFORM == FACE), TV>::type Sample(Int2Type<IFORM>, index_type s,
	        nTuple<NDIMS, TV> const & v) const
	{
		return Normal(s, v) * Volume(s);
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

	coordinates_type CoordinatesToCartesian(coordinates_type const &x) const
	{
		return x;
	}

	coordinates_type CoordinatesFromCartesian(coordinates_type const &x) const
	{
		return x;
	}

	template<typename TV>
	nTuple<NDIMS, TV> const& PushForward(coordinates_type const &x, nTuple<NDIMS, TV> const & v) const
	{
		return v;
	}
	template<typename TV>
	nTuple<NDIMS, TV> const& PullBack(coordinates_type const &x, nTuple<NDIMS, TV> const & v) const
	{
		return v;
	}

}
;

}  // namespace simpla

#endif /* GEOMETRY_EUCLIDEAN_H_ */
