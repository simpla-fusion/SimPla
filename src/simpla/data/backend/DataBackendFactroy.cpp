//
// Created by salmon on 17-3-6.
//
#include "DataBackendFactroy.h"
#include "../DataBackend.h"
#include "DataBackendHDF5.h"
#include "DataBackendLua.h"
#include "DataBackendMemory.h"
#include "DataBackendSAMRAI.h"

namespace simpla {
namespace data {
std::unique_ptr<DataBackend> CreateDataBackend(std::string const &url, std::string const &param) {
    if (url == "") { return std::make_unique<DataBackendMemory>(); }

    DataBackend *res = nullptr;
    std::string ext = "";
    size_type pos = url.find_last_of('.');
    if (pos != std::string::npos) { ext = url.substr(pos + 1); }
    if (ext == "lua") {
        return std::make_unique<DataBackendLua>(url, param);
    } else if (ext == "h5") {
        return std::make_unique<DataBackendHDF5>(url, param);
    } else if (ext == "samrai") {
        return std::make_unique<DataBackendSAMRAI>(url, param);
    } else {
        return std::make_unique<DataBackendMemory>(url, param);
    }
};
}  // namespace data{
}  // namespace simpla{