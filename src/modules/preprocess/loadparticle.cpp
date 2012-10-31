/*
 * loadparticle.cpp
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#include "loadparticle.h"
namespace simpla
{

namespace preprocess
{

LoadParticle::LoadParticle(BaseContext & d, const ptree & pt) :
		ctx(d)
{

}

LoadParticle::~LoadParticle()
{
}
TR1::function<void(void)> LoadParticle::Create(BaseContext* d, const ptree & pt)
{
	return TR1::bind(&ThisType::Eval,
			TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
}

void LoadParticle::Eval()
{
}

}  // namespace preprocess

}  // namespace simpla

