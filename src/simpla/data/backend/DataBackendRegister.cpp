//
// Created by salmon on 17-3-10.
//
#include "../DataBackend.h"
#include "../DataBackendMemory.h"
#include "DataBackendHDF5.h"
#include "DataBackendLua.h"
#include "DataBackendSAMRAI.h"

namespace simpla {
namespace data {
void DataBackendFactory::RegisterDefault() {
    Register<DataBackendSAMRAI>("samrai");
    Register<DataBackendLua>("lua");
    Register<DataBackendHDF5>("h5");
};
}  // namespace data {
}  // namespace simpla {