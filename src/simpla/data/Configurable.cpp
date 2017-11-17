//
// Created by salmon on 17-11-17.
//
#include "Configurable.h"
#include <boost/functional/hash.hpp>       //for uuid
#include <boost/uuid/uuid.hpp>             //for uuid
#include <boost/uuid/uuid_generators.hpp>  //for uuid
#include "DataEntry.h"
namespace simpla {
namespace data {

static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;

Configurable::Configurable() : m_db_(data::DataEntry::New(data::DataEntry::DN_TABLE)) {
    SetUUID(g_obj_hasher(g_uuid_generator()));
};
Configurable::Configurable(Configurable const &other) : Configurable() { m_db_->Set(other.m_db_); }
Configurable::Configurable(Configurable &&other) noexcept : m_db_(other.m_db_) { other.m_db_.reset(); }
Configurable::~Configurable() = default;
std::shared_ptr<const DataEntry> Configurable::db() const { return m_db_; }
std::shared_ptr<DataEntry> Configurable::db() { return m_db_; }

}  // namespace geometry
}  // namespace simpla