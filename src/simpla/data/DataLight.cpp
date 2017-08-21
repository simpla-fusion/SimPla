//
// Created by salmon on 17-8-20.
//

#include "DataLight.h"
namespace simpla {
namespace data {
std::shared_ptr<DataLight> DataLight::New() { return std::shared_ptr<DataLight>(new DataLight); }
}
}