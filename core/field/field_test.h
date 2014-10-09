/*
 * fetl_test.h
 *
 *  created on: 2014-2-20
 *      Author: salmon
 */

#ifndef FIELD_TEST_H_
#define FIELD_TEST_H_
#include <gtest/gtest.h>

#include "field.h"
using namespace simpla;
#include <vector>
int main(int argc, char **argv)
{
	_Field<double> f;
}

//#include "../utilities/log.h"
//#include "../utilities/pretty_stream.h"
//
//using namespace simpla;
//
//class Domain;
//class Container;
//
//typedef _Field<Domain, Container> Field;
//
//class TestFIELD: public testing::TestWithParam<
//		std::tuple<typename Domain::coordinates_type,
//				typename Domain::coordinates_type,
//				nTuple<Domain::NDIMS, size_t>, nTuple<Domain::NDIMS, Real> > >
//{
//
//protected:
//	void SetUp()
//	{
//		LOGGER.set_stdout_visable_level(LOG_INFORM);
//		auto param = GetParam();
//
//		xmin = std::get<0>(param);
//
//		xmax = std::get<1>(param);
//
//		dims = std::get<2>(param);
//
//		K_real = std::get<3>(param);
//
//		SetDefaultValue(&default_value);
//
//		for (int i = 0; i < NDIMS; ++i)
//		{
//			if (dims[i] <= 1 || (xmax[i] <= xmin[i]))
//			{
//				dims[i] = 1;
//				K_real[i] = 0.0;
//				xmax[i] = xmin[i];
//			}
//		}
//
//		mesh.set_dimensions(dims);
//		mesh.set_extents(xmin, xmax);
//
//		mesh.update();
//
//	}
//public:
//
//	typedef Domain domain_type;
//	typedef Real value_type;
//	typedef domain_type::scalar_type scalar_type;
//	typedef domain_type::iterator iterator;
//	typedef domain_type::coordinates_type coordinates_type;
//
//	domain_type mesh;
//
//	static constexpr unsigned int NDIMS = domain_type::NDIMS;
//
//	nTuple<NDIMS, Real> xmin;
//
//	nTuple<NDIMS, Real> xmax;
//
//	nTuple<NDIMS, size_t> dims;
//
//	nTuple<3, Real> K_real; // @NOTE must   k = n TWOPI, period condition
//
//	nTuple<3, scalar_type> K_imag;
//
//	value_type default_value;
//
//	template<typename T>
//	void SetDefaultValue(T* v)
//	{
//		*v = 1;
//	}
//	template<typename T>
//	void SetDefaultValue(std::complex<T>* v)
//	{
//		T r;
//		SetDefaultValue(&r);
//		*v = std::complex<T>();
//	}
//
//	template<unsigned int N, typename T>
//	void SetDefaultValue(nTuple<N, T>* v)
//	{
//		for (int i = 0; i < N; ++i)
//		{
//			(*v)[i] = i;
//		}
//	}
//
//	virtual ~TestFIELD()
//	{
//
//	}
//
//};

#endif /* FIELD_TEST_H_ */
