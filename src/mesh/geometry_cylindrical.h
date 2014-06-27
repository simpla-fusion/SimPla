/*
 * geometry_cylindrical.h
 *
 *  Created on: 2014年3月13日
 *      Author: salmon
 */

#ifndef GEOMETRY_CYLINDRICAL_H_
#define GEOMETRY_CYLINDRICAL_H_

#include <iostream>
#include <utility>
#include <cmath>
#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"

namespace simpla
{

/***
 *
 *   R Z theta
 */
template<typename TTopology, bool BEnableSpectralMethod = false, int IZAxis = 2>
struct CylindricalGeometry: public TTopology
{
	typedef TTopology topology_type;

	static constexpr bool enable_spectral_method = BEnableSpectralMethod;

	static constexpr unsigned int ZAxis = IZAxis;
	static constexpr unsigned int RAxis = (IZAxis + 1) % 3;
	static constexpr unsigned int ThetaAxis = (IZAxis + 2) % 3;

	typedef CylindricalGeometry<topology_type, enable_spectral_method, ZAxis> this_type;

	static constexpr int NDIMS = topology_type::NDIMS;

	typedef typename topology_type::coordinates_type coordinates_type;
	typedef typename topology_type::index_type index_type;
	typedef typename std::conditional<enable_spectral_method, std::complex<Real>, Real>::type scalar_type;
	typedef typename topology_type::compact_index_type compact_index_type;
	typedef typename topology_type::iterator iterator;

	typedef nTuple<NDIMS, Real> vector_type;
	typedef nTuple<NDIMS, Real> covector_type;

	CylindricalGeometry(this_type const & rhs) = delete;

	CylindricalGeometry() :
			topology_type()
	{

	}
	template<typename TDict>
	CylindricalGeometry(TDict const & dict) :
			topology_type(dict)
	{
		Load(dict);
	}

	~CylindricalGeometry()
	{
	}

	//***************************************************************************************************
	// Geometric properties
	// Metric
	//***************************************************************************************************

	Real dt_ = 0.0;
	Real time0_ = 0.0;
	// Time
	void NextTimeStep()
	{
		topology_type::NextTimeStep();
	}
	Real GetTime() const
	{
		return static_cast<double>(topology_type::GetClock()) * dt_ + time0_;
	}

	Real GetDt() const
	{
		return dt_;
	}

	coordinates_type xmin_ =
	{ 0, 0, 0 };

	coordinates_type xmax_ =
	{ 1, 1, 1 };

	coordinates_type inv_length_ =
	{ 1.0, 1.0, 1.0 };

	coordinates_type length_ =
	{ 1.0, 1.0, 1.0 };

	coordinates_type shift_ =
	{ 0, 0, 0 };

	template<typename TDict, typename ...Others>
	void Load(TDict const & dict, Others &&...others)
	{
		try
		{
			if (dict["Min"] && dict["Max"])
			{
				LOGGER << "Load CylindricalGeometry ";

				SetExtents(

				dict["Min"].template as<nTuple<NDIMS, Real>>(),

				dict["Max"].template as<nTuple<NDIMS, Real>>());
			}

			dt_ = dict["dt"].template as<Real>();

			topology_type::Load(dict, std::forward<Others >(others)...);

		} catch (...)
		{
			PARSER_ERROR("Configure CylindricalGeometry error!");
		}
	}

	std::string Save(std::string const &path) const
	{
		std::stringstream os;

		os << "\tMin = " << xmin_ << " , " << "Max  = " << xmax_ << ", " << " dt  = " << dt_ << ", "

		<< topology_type::Save(path);

		return os.str();
	}
	template<typename ...Others>
	inline void SetExtents(nTuple<NDIMS, Real> const & pmin, nTuple<NDIMS, Real> const & pmax, Others&& ... others)
	{

		topology_type::SetDimensions(std::forward<Others >(others)...);
		SetExtents(pmin, pmax);
	}

	void SetExtents(nTuple<NDIMS, Real> const & pmin, nTuple<NDIMS, Real> const & pmax,
			nTuple<NDIMS, Real> const & dims)
	{
		topology_type::SetDimensions(dims);
		SetExtents(pmin, pmax);
	}
	void SetExtents(nTuple<NDIMS, Real> const & pmin, nTuple<NDIMS, Real> const & pmax)
	{

		if (pmin[RAxis] < EPSILON)
		{
			RUNTIME_ERROR(" illegal configure: Cylindrical R_min=0, xmin=" + ToString(pmin));
		}

		auto dims = topology_type::GetDimensions();

		for (int i = 0; i < NDIMS; ++i)
		{
			xmin_[i] = pmin[i];

			shift_[i] = xmin_[i];

			if ((pmax[i] - pmin[i]) < EPSILON || dims[i] <= 1)
			{

				xmax_[i] = xmin_[i];

				inv_length_[i] = 0.0;

				length_[i] = 0.0;

			}
			else
			{
				xmax_[i] = pmax[i];

				length_[i] = (xmax_[i] - xmin_[i]);

				if (i == ThetaAxis && length_[i] > TWOPI)
				{
					xmax_[i] = xmin_[i] + TWOPI;

					length_[i] = TWOPI;
				}

				inv_length_[i] = 1.0 / length_[i];

			}
		}

		UpdateVolume();
	}
	inline std::pair<coordinates_type, coordinates_type> GetExtents() const
	{
		return std::move(std::make_pair(xmin_, xmax_));
	}

	inline coordinates_type GetDx(compact_index_type s = 0UL) const
	{

		coordinates_type res;

		auto d = topology_type::GetDx();

		for (int i = 0; i < NDIMS; ++i)
		{
			res[i] = length_[i] * d[i];
		}

		return std::move(res);
	}

	template<typename ... Args>
	inline coordinates_type GetCoordinates(Args && ... args) const
	{
		return CoordinatesFromTopology(topology_type::GetCoordinates(std::forward<Args >(args)...));
	}

	coordinates_type CoordinatesFromTopology(coordinates_type const &x) const
	{

		return coordinates_type(
		{

		x[0] * length_[0] + shift_[0],

		x[1] * length_[1] + shift_[1],

		x[2] * length_[2] + shift_[2]

		});

	}
	coordinates_type CoordinatesToTopology(coordinates_type const &x) const
	{
		return coordinates_type(
		{

		(x[0] - shift_[0]) * inv_length_[0],

		(x[1] - shift_[1]) * inv_length_[1],

		(x[2] - shift_[2]) * inv_length_[2]

		});

	}
	template<typename ... Args>
	inline coordinates_type CoordinatesLocalToGlobal(Args && ... args) const
	{
		return CoordinatesFromTopology(topology_type::CoordinatesLocalToGlobal(std::forward<Args >(args)...));
	}

	std::tuple<compact_index_type, coordinates_type> CoordinatesGlobalToLocal(coordinates_type x,
			typename topology_type::compact_index_type shift = 0UL) const
	{
		return std::move(topology_type::CoordinatesGlobalToLocal(std::move(CoordinatesToTopology(x)), shift));
	}
	coordinates_type InvMapTo(coordinates_type const &r, unsigned int CartesianZAxis = 2) const
	{
		coordinates_type x;

		x[(CartesianZAxis + 1) % 3] = r[RAxis] * std::cos(r[ThetaAxis]);
		x[(CartesianZAxis + 2) % 3] = r[RAxis] * std::sin(r[ThetaAxis]);
		x[(CartesianZAxis + 3) % 3] = r[ZAxis];

		return std::move(x);
	}

	coordinates_type MapTo(coordinates_type const &x, unsigned int CartesianZAxis = 2) const
	{
		coordinates_type r;

		r[ZAxis] = x[CartesianZAxis];
		r[RAxis] = std::sqrt(
				x[(CartesianZAxis + 1) % 3] * x[(CartesianZAxis + 1) % 3]
						+ x[(CartesianZAxis + 2) % 3] * x[(CartesianZAxis + 2) % 3]);
		r[ThetaAxis] = std::atan2(x[(CartesianZAxis + 2) % 3], x[(CartesianZAxis + 1) % 3]);

		return r;
	}

	/**
	 *
	 *   transform vector from Cartesian to Cylindrical
	 *
	 * @param x (x,y,z)
	 *         u[XAixs] \partial_x +  u[YAixs] \partial_y + u[ZAixs] \partial_z
	 * @param ZAxisOfVector
	 * @return  (r,z,theta)
	 *          v = v[RAixs] \partial_r +  v[1] 1/x[RAxis] \partial_theta + v[ZAixs] \partial_z
	 *
	 *
	 *
	 *
	 *
	 *     theta   y   r
	 *          \  |  /
	 *           \ | /
	 *            \|/------x
	 *          y  /
	 *          | /
	 *          |/)theta
	 *          0------x
	 *
	 *
	 */

	template<typename TV>
	std::tuple<coordinates_type, nTuple<NDIMS, TV> > PushForward(
			std::tuple<coordinates_type, nTuple<NDIMS, TV> > const & Z, unsigned int CartesianZAxis = 2) const
	{
		coordinates_type r = MapTo(std::get<0>(Z), CartesianZAxis);

		auto const & v = std::get<1>(Z);

		nTuple<NDIMS, TV> u;

		Real c = std::cos(r[ThetaAxis]), s = std::sin(r[ThetaAxis]);

		u[ZAxis] = v[CartesianZAxis % 3];

		u[RAxis] = v[(CartesianZAxis + 1) % 3] * c + v[(CartesianZAxis + 2) % 3] * s;

		u[ThetaAxis] = (-v[(CartesianZAxis + 1) % 3] * s + v[(CartesianZAxis + 2) % 3] * c) / r[RAxis];

		return std::move(std::make_tuple(r, u));
	}

	/**
	 *
	 *  transform vector  from    Cylindrical to Cartesian
	 * @param r z theta
	 *  u = u[RAixs] \partial_r +  u[1]  r[RAxis] \partial_theta + u[ZAixs] \partial_z
	 * @param ZAxisOfVector
	 * @return  x, v = v[XAixs] \partial_x +  v[YAixs] \partial_y + v[ZAixs] \partial_z
	 */
	template<typename TV>
	std::tuple<coordinates_type, nTuple<NDIMS, TV> > PullBack(
			std::tuple<coordinates_type, nTuple<NDIMS, TV> > const & R, unsigned int CartesianZAxis = 2) const
	{
		auto const & r = std::get<0>(R);
		auto const & u = std::get<1>(R);

		Real c = std::cos(r[ThetaAxis]), s = std::sin(r[ThetaAxis]);

		nTuple<NDIMS, TV> v;

		v[(CartesianZAxis + 1) % 3] = u[RAxis] * c - u[ThetaAxis] * r[RAxis] * s;
		v[(CartesianZAxis + 2) % 3] = u[RAxis] * s + u[ThetaAxis] * r[RAxis] * c;
		v[(CartesianZAxis + 3) % 3] = u[ZAxis];

		return std::move(std::make_tuple(InvMapTo(r), v));
	}

	auto Select(unsigned int iform, coordinates_type const & xmin, coordinates_type const & xmax) const
	DECL_RET_TYPE((topology_type::Select(iform, CoordinatesToTopology(xmin),CoordinatesToTopology(xmax))))

	template<typename ...Args>
	auto Select(unsigned int iform, Args && ...args) const
	DECL_RET_TYPE((topology_type::Select(iform,std::forward<Args >(args)...)))

	template<typename TV>
	TV const& Normal(index_type s, TV const & v) const
	{
		return v;
	}

	template<typename TV>
	TV const& Normal(index_type s, nTuple<3, TV> const & v) const
	{
		return v[topology_type::ComponentNum(s)];
	}

	template<typename TV>
	TV Sample(index_type s, TV const &v) const
	{
		return v;
	}

	template<typename TV>
	TV Sample(index_type s, nTuple<3, TV> const &v) const
	{
		return PushForward(s, v)[topology_type::ComponentNum(s)];
	}

	template<int IFORM, typename TV>
	typename std::enable_if<(IFORM == EDGE || IFORM == FACE), TV>::type Sample(Int2Type<IFORM>, index_type s,
			nTuple<NDIMS, TV> const & v) const
	{
		return Normal(s, v);
	}

	Real volume_[8] =
	{ 1, // 000
			1, //001
			1, //010
			1, //011
			1, //100
			1, //101
			1, //110
			1  //111
			};
	Real inv_volume_[8] =
	{ 1, 1, 1, 1, 1, 1, 1, 1 };

	Real dual_volume_[8] =
	{ 1, 1, 1, 1, 1, 1, 1, 1 };

	Real inv_dual_volume_[8] =
	{ 1, 1, 1, 1, 1, 1, 1, 1 };

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
	void UpdateVolume()
	{

		for (int i = 0; i < NDIMS; ++i)
		{

			if ((xmax_[i] - xmin_[i]) < EPSILON)
			{

				volume_[1UL << (NDIMS - i - 1)] = 1.0;

				dual_volume_[7 - (1UL << (NDIMS - i - 1))] = 1.0;

				inv_volume_[1UL << (NDIMS - i - 1)] = 1.0;

				inv_dual_volume_[7 - (1UL << (NDIMS - i - 1))] = 1.0;

			}
			else
			{

				volume_[1UL << (NDIMS - i - 1)] = length_[i];

				dual_volume_[7 - (1UL << (NDIMS - i - 1))] = length_[i];

				inv_volume_[1UL << (NDIMS - i - 1)] = inv_length_[i];

				inv_dual_volume_[7 - (1UL << (NDIMS - i - 1))] = inv_length_[i];

			}
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
		//		volume_[1] /* 001 */= dx_[0];
		//		volume_[2] /* 010 */= dx_[1];
		//		volume_[4] /* 100 */= dx_[2];

		volume_[3] /* 011 */= volume_[1] * volume_[2];
		volume_[5] /* 101 */= volume_[4] * volume_[1];
		volume_[6] /* 110 */= volume_[2] * volume_[4];

		volume_[7] /* 111 */= volume_[1] * volume_[2] * volume_[4];

		dual_volume_[7] = 1;
		//		dual_volume_[6] /* 001 */= dx_[0];
		//		dual_volume_[5] /* 010 */= dx_[1];
		//		dual_volume_[3] /* 100 */= dx_[2];

		dual_volume_[4] /* 011 */= dual_volume_[6] * dual_volume_[5];
		dual_volume_[2] /* 101 */= dual_volume_[3] * dual_volume_[6];
		dual_volume_[1] /* 110 */= dual_volume_[5] * dual_volume_[3];

		dual_volume_[0] /* 111 */= dual_volume_[6] * dual_volume_[5] * dual_volume_[3];

		inv_volume_[0] = 1;
		//		inv_volume_[1] /* 001 */= inv_dx_[0];
		//		inv_volume_[2] /* 010 */= inv_dx_[1];
		//		inv_volume_[4] /* 100 */= inv_dx_[2];

		inv_volume_[3] /* 011 */= inv_volume_[1] * inv_volume_[2];
		inv_volume_[5] /* 101 */= inv_volume_[4] * inv_volume_[1];
		inv_volume_[6] /* 110 */= inv_volume_[2] * inv_volume_[4];

		inv_volume_[7] /* 111 */= inv_volume_[1] * inv_volume_[2] * inv_volume_[4];

		inv_dual_volume_[7] = 1;
		//		inv_dual_volume_[6] /* 001 */= inv_dx_[0];
		//		inv_dual_volume_[5] /* 010 */= inv_dx_[1];
		//		inv_dual_volume_[3] /* 100 */= inv_dx_[2];

		inv_dual_volume_[4] /* 011 */= inv_dual_volume_[6] * inv_dual_volume_[5];
		inv_dual_volume_[2] /* 101 */= inv_dual_volume_[3] * inv_dual_volume_[6];
		inv_dual_volume_[1] /* 110 */= inv_dual_volume_[5] * inv_dual_volume_[3];

		inv_dual_volume_[0] /* 111 */= inv_dual_volume_[6] * inv_dual_volume_[5] * inv_dual_volume_[3];

	}

//	scalar_type Volume(compact_index_type s) const
//	{
//		return topology_type::Volume(s) * volume_[topology_type::NodeId(s)];
//	}
//	scalar_type InvVolume(compact_index_type s) const
//	{
//		return topology_type::InvVolume(s) * inv_volume_[topology_type::NodeId(s)];
//	}
//
//	scalar_type DualVolume(compact_index_type s) const
//	{
//		return topology_type::DualVolume(s) * dual_volume_[topology_type::NodeId(s)];
//	}
//	scalar_type InvDualVolume(compact_index_type s) const
//	{
//		return topology_type::InvDualVolume(s) * inv_dual_volume_[topology_type::NodeId(s)];
//	}

	Real HodgeStarVolumeScale(compact_index_type s) const
	{
		return 1.0;
	}

	Real CellVolume(compact_index_type s) const
	{
		return topology_type::CellVolume(s) * volume_[1] * volume_[2] * volume_[4] * GetCoordinates(s)[RAxis];
	}

	scalar_type Volume(compact_index_type s) const
	{
		unsigned int n = topology_type::NodeId(s);
		return topology_type::Volume(s) * volume_[n]
				* (((n & (1UL << (NDIMS - ThetaAxis - 1))) > 0) ? GetCoordinates(s)[RAxis] : 1.0);
	}

	scalar_type InvVolume(compact_index_type s) const
	{
		unsigned int n = topology_type::NodeId(s);
		return topology_type::InvVolume(s) * inv_volume_[n]
				/ (((n & (1UL << (NDIMS - ThetaAxis - 1))) > 0) ? GetCoordinates(s)[RAxis] : 1.0);
	}

	scalar_type DualVolume(compact_index_type s) const
	{
		unsigned int n = topology_type::NodeId(topology_type::Dual(s));
		return topology_type::DualVolume(s) * volume_[n]
				* (((n & (1UL << (NDIMS - ThetaAxis - 1))) > 0) ? GetCoordinates(s)[RAxis] : 1.0);
	}
	scalar_type InvDualVolume(compact_index_type s) const
	{
		unsigned int n = topology_type::NodeId(topology_type::Dual(s));
		return topology_type::InvDualVolume(s) * inv_volume_[n]
				/ (((n & (1UL << (NDIMS - ThetaAxis - 1))) > 0) ? GetCoordinates(s)[RAxis] : 1.0);
	}

}
;

}  // namespace simpla

#endif /* GEOMETRY_CYLINDRICAL_H_ */
