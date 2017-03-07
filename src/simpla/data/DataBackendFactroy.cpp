//
// Created by salmon on 17-3-6.
//
#include "DataBackendFactroy.h"
#include "DataBackend.h"
#include "DataBackendLua.h"
#include "DataBackendMemory.h"
namespace simpla {
namespace data {
DataBackend *CreateDataBackendFromFile(std::string const &url) {
    DataBackend *res = nullptr;
    std::string ext = "";
    size_type pos = url.find_last_of('.');
    if (pos != std::string::npos) { ext = url.substr(pos + 1); }
    if (ext == "lua") {
        res = new DataBackendLua(url);
    } else {
        res = new DataBackendMemory(url);
    }
    return res;
};
}  // namespace data{
}  // namespace simpla{