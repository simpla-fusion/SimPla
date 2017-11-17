/**
 * @file object.cpp
 * @author salmon
 * @date 2015-12-16.
 */

#include "simpla/SIMPLA_config.h"

#include "SPObject.h"

//#include <simpla/parallel/MPIComm.h>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <iomanip>
#include <ostream>

#include "simpla/utilities/Log.h"
#include "simpla/utilities/type_cast.h"

namespace simpla {
struct SPObject::pimpl_s {
    bool m_is_initialized_ = false;
};

static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;

SPObject::SPObject() = default;
SPObject::SPObject(SPObject const &other) = default;
SPObject::~SPObject() = default;

std::shared_ptr<SPObject> SPObject::Create(std::string const &key) { return Factory<SPObject>::Create(key); };
std::shared_ptr<SPObject> SPObject::Create(std::shared_ptr<data::DataEntry> const &tdb) {
    auto res = Factory<SPObject>::Create(tdb->GetValue<std::string>("_TYPE_"));
    res->Deserialize(tdb);
    return res;
};

std::ostream &operator<<(std::ostream &os, SPObject const &obj) {
    std::cout << *obj.Serialize() << std::endl;
    return os;
}
std::istream &operator>>(std::istream &is, SPObject &obj) {
    obj.Deserialize(data::DataEntry::New(data::DataEntry::DN_TABLE, std::string(std::istreambuf_iterator<char>(is), {})));
    return is;
}

std::ostream &operator<<(std::ostream &os, std::shared_ptr<const SPObject> const &obj) {
    if (obj != nullptr) {
        os << *obj;
    } else {
        os << "<NULL OBJECT>";
    }
    return os;
}
std::istream &operator<<(std::istream &is, std::shared_ptr<SPObject> const &obj) {
    if (obj != nullptr) { is >> *obj; }
    return is;
}

}  // namespace simpla
