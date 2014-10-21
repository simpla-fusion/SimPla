/*
 * graph_algorithm.h
 *
 *  created on: 2014-5-27
 *      Author: salmon
 */

#ifndef GRAPH_ALGORITHM_H_
#define GRAPH_ALGORITHM_H_

#include <utility>

#include "../utilities/ntuple.h"

namespace simpla
{

/**
 * \ingroup  Numeric
 * \brief calculate the overlap of two  hexahedron
 */
template<unsigned int N, typename T>
void Clipping(nTuple<T,N> const&xmin, nTuple<T,N> const&xcount, nTuple<T,N> const&ymin, nTuple<T,N> const& ycount,
        nTuple<T,N>* rmin, nTuple<T,N>* rcount)
{

	for (int i = 0; i < N; ++i)
	{

		rmin[i] = (ymin[i] > xmin[i]) ? ymin[i] : xmin[i];

		rcount[i] =
		        (ymin[i] + ycount[i] < xmin[i] + xcount[i]) ?
		                ymin[i] + ycount[i] - rmin[i] : xmin[i] + xcount[i] - rmin[i];
	}

}
}  // namespace simpla

#endif /* GRAPH_ALGORITHM_H_ */
