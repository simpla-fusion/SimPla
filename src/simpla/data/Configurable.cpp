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

Configurable::Configurable()
    : m_id_(g_obj_hasher(g_uuid_generator())), m_db_(data::DataEntry::New(data::DataEntry::DN_TABLE)) {
    m_db_->SetValue("UUID", GetUUID());
};
Configurable::Configurable(Configurable const &other) : Configurable() { m_db_->Set(other.m_db_); }
Configurable::~Configurable() = default;
id_type Configurable::GetUUID() const { return m_id_; }
void Configurable::SetName(std::string const &s) { m_db_->SetValue("Name", s); }
std::string Configurable::GetName() const { return m_db_->GetValue<std::string>("Name", std::string("unnamed")); }
std::shared_ptr<const DataEntry> Configurable::db() const { return m_db_; }
std::shared_ptr<DataEntry> Configurable::db() { return m_db_; }
void Configurable::db(std::shared_ptr<DataEntry> const &) {}

}  // namespace geometry
}  // namespace simpla