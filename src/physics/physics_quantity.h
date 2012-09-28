/*
 * physics_quantity.h
 *
 *  Created on: 2012-3-6
 *      Author: salmon
 */

#ifndef PHYSICS_QUANTITY_H_
#define PHYSICS_QUANTITY_H_

namespace physics
{
template<typename T>
bool physics_equal(T const & lhs, T const & rhs)
{
	return (lhs == rhs);
}

bool physics_equal(double const & lhs, double const & rhs)
{ // NOTE: the relative standard uncertainty of  most physical constants are around 1e-8
	return (fabs(2.0 * (lhs - rhs) / (lhs + rhs)) < 1.0e-10);
}

template<int IS, int I0, int I1, int I2, int I3, int I4, int I5, int I6,
		typename TV = double>
struct PhysicalQuantity
{
	TV value_;

	typedef TV ValueType;

	typedef PhysicalQuantity<IS, I0, I1, I2, I3, I4, I5, I6, ValueType> ThisType;

	PhysicalQuantity() :
			value_(0.0)
	{
	}

	PhysicalQuantity(ValueType const &v) :
			value_(v)
	{
	}

	PhysicalQuantity(ThisType const & rhs) :
			value_(rhs.value_)
	{
	}

	ThisType & operator=(ThisType const & rhs)
	{
		value_ = rhs.value_;
		return (*this);
	}

	inline const ValueType value() const
	{
		return (value_);
	}
	inline ValueType value()
	{
		return (value_);
	}

	inline bool operator==(ThisType const & rhs) const
	{
		return (physics_equal(value_, rhs.value_));
	}

};
template<typename TV> struct PhysicalQuantityTraits;

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TV>
struct PhysicalQuantityTraits<
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> >
{
	typedef PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> ValueType;
	static inline ValueType get_value(TV const & v)
	{
		return (ValueType(v));
	}
};

// dimensionless quantity
template<int US, typename TV>
struct PhysicalQuantityTraits<PhysicalQuantity<US, 0, 0, 0, 0, 0, 0, 0, TV> >
{
	typedef TV ValueType;
	static inline ValueType get_value(ValueType const & v)
	{
		return ((v));
	}
};

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TL, int R0, int R1, int R2, int R3, int R4, int R5, int R6,
		typename TR>
inline typename PhysicalQuantityTraits<
		PhysicalQuantity<US, L0 + R0, L1 + R1, L2 + R2, L3 + R3, L4 + R4,
				L5 + R5, L6 + R6, decltype(TL()*TR())> >::ValueType //
operator*(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL> const & lhs
		, PhysicalQuantity<US, R0, R1, R2, R3, R4, R5, R6, TR> const & rhs)
{
	return (PhysicalQuantityTraits<
			PhysicalQuantity<US, L0 + R0, L1 + R1, L2 + R2, L3 + R3, L4 + R4,
					L5 + R5, L6 + R6, decltype(TL()*TR())> >::get_value(
			lhs.value() * rhs.value()));
}

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TL, int R0, int R1, int R2, int R3, int R4, int R5, int R6,
		typename TR>
inline typename PhysicalQuantityTraits<
		PhysicalQuantity<US, L0 - R0, L1 - R1, L2 - R2, L3 - R3, L4 - R4,
				L5 - R5, L6 - R6, decltype(TL()/TR())> >::ValueType //
operator/(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL> const & lhs
		, PhysicalQuantity<US, R0, R1, R2, R3, R4, R5, R6, TR> const & rhs)
{
	return (PhysicalQuantityTraits<
			PhysicalQuantity<US, L0 - R0, L1 - R1, L2 - R2, L3 - R3, L4 - R4,
					L5 - R5, L6 - R6, decltype(TL()/TR())>>::get_value(
			lhs.value() / rhs.value()));
}

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TV>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>  //
operator+(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> const & lhs
		, PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> const & rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>(
			lhs.value() + rhs.value()));
}
template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TV>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>  //
operator-(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> const & lhs
		, PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> const & rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>(
			lhs.value() - rhs.value()));
}
template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TL, typename TR>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL()*TR())>  //
operator*(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL> const & lhs
		, TR const &rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL()*TR())>(
			lhs.value() * rhs));
}
template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TV>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>  //
operator*(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV> const & lhs
		, TV const &rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TV>(
			lhs.value() * rhs));
}
template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TR>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR>  //
operator*(TR const & lhs,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR> const &rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR>(
			lhs * rhs.value()));
}
template<int US, typename TL, int L0, int L1, int L2, int L3, int L4, int L5,
		int L6, typename TR>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL()*TR())>  //
operator*(TL const & lhs,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR> const &rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL()*TR())>(
			lhs * rhs.value()));
}

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TL, typename TR>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL(),TR())>  //
operator/(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL> const & lhs
		, TR rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, decltype(TL(),TR())>(
			lhs.value() / rhs));
}
template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TL>
inline PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL>  //
operator/(PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL> const & lhs
		, TL const & rhs)
{
	return (PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TL>(
			lhs.value() / rhs));
}

template<int US, typename TL, int L0, int L1, int L2, int L3, int L4, int L5,
		int L6, typename TR>
inline PhysicalQuantity<US, -L0, -L1, -L2, -L3, -L4, -L5, -L6,
		decltype(TL()/TR())>  //
operator/(TL lhs,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR> const & rhs)
{
	return (PhysicalQuantity<US, -L0, -L1, -L2, -L3, -L4, -L5, -L6,
			decltype(TL()/TR())>(lhs / rhs.value()));
}

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6,
		typename TR>
inline PhysicalQuantity<US, -L0, -L1, -L2, -L3, -L4, -L5, -L6, TR>  //
operator/(TR lhs,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6, TR> const & rhs)
{
	return (PhysicalQuantity<US, -L0, -L1, -L2, -L3, -L4, -L5, -L6, TR>(
			lhs / rhs.value()));
}

namespace units
{
template<int US> struct Symbol;
} // namespace units

template<int US, int L0, int L1, int L2, int L3, int L4, int L5, int L6>
std::ostream &
operator<<(std::ostream & oss,
		PhysicalQuantity<US, L0, L1, L2, L3, L4, L5, L6> const & rhs)
{

	oss << rhs.value() << " [";

	if (L0 != 0)
	{
		oss << " " << units::Symbol<US>::str[0];
		if (L0 != 1)
		{
			oss << "^" << L0;
		}
	}

	if (L1 != 0)
	{
		oss << " " << units::Symbol<US>::str[1];

		if (L1 != 1)
		{
			oss << "^" << L1;
		}
	}

	if (L2 != 0)
	{
		oss << " " << units::Symbol<US>::str[2];
		if (L2 != 1)
		{
			oss << "^" << L2;
		}
	}
	if (L3 != 0)
	{
		oss << " " << units::Symbol<US>::str[3];
		if (L3 != 1)
		{
			oss << "^" << L3;
		}
	}
	if (L4 != 0)
	{
		oss << " " << units::Symbol<US>::str[4];
		if (L4 != 1)
		{
			oss << "^" << L4;
		}
	}
	if (L5 != 0)
	{
		oss << " " << units::Symbol<US>::str[5];
		if (L4 != 1)
		{
			oss << "^" << L4;
		}
	}
	if (L6 != 0)
	{
		oss << " " << units::Symbol<US>::str[6];
		if (L6 != 1)
		{
			oss << "^" << L6;
		}
	}
	oss << " ]";
	return oss;

}

} // namespace Physics

#endif /* PHYSICS_QUANTITY_H_ */
