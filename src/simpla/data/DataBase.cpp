//
// Created by salmon on 17-3-9.
//

#include "DataBase.h"
#include <iomanip>
#include <string>

#include "db/DataBaseHDF5.h"
#include "db/DataBaseLua.h"
#include "db/DataBaseMemory.h"
#include "db/DataBaseStdIO.h"
#include "simpla/utilities/Factory.h"
#include "simpla/utilities/ParsingURI.h"
namespace simpla {
namespace data {

std::shared_ptr<DataBase> DataBase::New(std::string const& s) {
    if (DataBase::s_num_of_pre_registered_ == 0) { RUNTIME_ERROR << "No database is registered!" << s << std::endl; }
    std::string uri = s.empty() ? "mem://" : s;

    std::string scheme;
    std::string path;
    std::string authority;
    std::string query;
    std::string fragment;

    std::tie(scheme, authority, path, query, fragment) = ParsingURI(uri);
    auto res = Factory<DataBase>::Create(scheme);
    ASSERT(res != nullptr);
    if (SP_SUCCESS != res->Connect(authority, path, query, fragment)) {
        RUNTIME_ERROR << "Fail to connect  Data Backend [ " << scheme << " : " << authority << path << " ]"
                      << std::endl;
    }

    return res;
};

}  // namespace data {
}  // namespace simpla {