/*
 * preprocess.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PREPROCESS_H_
#define PREPROCESS_H_
#include "include/simpla_defs.h"
#include "utilities/properties.h"
#include "engine/basecontext.h"

#include "datastruct/ndarray.h"
#include "datastruct/pool.h"
#include "datastruct/compound.h"

namespace simpla
{
namespace preprocess
{
TR1::shared_ptr<NdArray> LoadField(BaseContext * ctx, const ptree & pt);

TR1::shared_ptr<Pool> LoadPool(BaseContext * ctx, const ptree & pt);

TR1::shared_ptr<CompoundObject> LoadCompound(BaseContext * ctx,
		const ptree & pt);

void Preprocess(BaseContext * ctx, const ptree & pt);
} // namespace preprocess
}  // namespace simpla

#endif /* PREPROCESS_H_ */
