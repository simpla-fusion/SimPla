/*
 * graph_algorithm.h
 *
 *  Created on: 2014年5月27日
 *      Author: salmon
 */

#ifndef GRAPH_ALGORITHM_H_
#define GRAPH_ALGORITHM_H_

#include <utility>

#include "../utilities/ntuple.h"

namespace simpla
{

/**
 *  \defgroup  Algorithm Algorithm
 *
 *
 */

/**
 * \ingroup  Geometry
 * \brief calculate the overlap of two  hexahedron
 */
template<int N, typename T>
void Clipping(nTuple<N, T> const&xmin, nTuple<N, T> const&xcount, nTuple<N, T> const&ymin, nTuple<N, T> const& ycount,
        nTuple<N, T>* rmin, nTuple<N, T>* rcount)
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
