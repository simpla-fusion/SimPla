/*
 * fetl_test.h
 *
 *  Created on: 2014年2月20日
 *      Author: salmon
 */

#ifndef FETL_TEST_H_
#define FETL_TEST_H_

#include "fetl.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

#include "../mesh/mesh.h"

using namespace simpla;

template<typename T>
void SetDefaultValue(T* v)
{
	*v = 1;
}
template<typename T>
void SetDefaultValue(std::complex<T>* v)
{
	T r;
	SetDefaultValue(&r);
	*v = std::complex<T>();
}

template<int N, typename T>
void SetDefaultValue(nTuple<N, T>* v)
{
	for (int i = 0; i < N; ++i)
	{
		(*v)[i] = i;
	}
}
#endif /* FETL_TEST_H_ */
