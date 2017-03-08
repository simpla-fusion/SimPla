//
// Created by salmon on 17-3-6.
//
#include "DataBackendFactroy.h"
#include "DataBackend.h"
#include "DataBackendLua.h"
#include "DataBackendMemory.h"
namespace simpla {
namespace data {
std::unique_ptr<DataBackend> CreateDataBackendFromFile(std::string const &url, std::string const &status) {
    if (url == "") { return std::make_unique<DataBackendMemory>(); }

    DataBackend *res = nullptr;
    std::string ext = "";
    size_type pos = url.find_last_of('.');
    if (pos != std::string::npos) { ext = url.substr(pos + 1); }
    if (ext == "lua") {
        return std::make_unique<DataBackendLua>(url, status);
    } else {
        return std::make_unique<DataBackendMemory>(url, status);
    }
};
}  // namespace data{
}  // namespace simpla{