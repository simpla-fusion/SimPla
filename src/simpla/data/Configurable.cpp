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
PropertyObserver::PropertyObserver(Configurable *host, std::string const &key) : m_host_(host), m_name_(key) {
    host->Attach(this);
}

PropertyObserver::~PropertyObserver() { m_host_->Detach(this); }

static boost::hash<boost::uuids::uuid> g_obj_hasher;
static boost::uuids::random_generator g_uuid_generator;

Configurable::Configurable() : m_db_(data::DataEntry::New(data::DataEntry::DN_TABLE)) {
    SetUUID(g_obj_hasher(g_uuid_generator()));
};

Configurable::Configurable(Configurable const &other) : Configurable() { m_db_->Set(other.m_db_); }

Configurable::~Configurable() = default;

void Configurable::Detach(PropertyObserver *attr) { m_observers_.erase(attr->GetName()); }

void Configurable::Attach(PropertyObserver *attr) { m_observers_.emplace(attr->GetName(), attr); }

void Configurable::Push(std::shared_ptr<const DataEntry> const &cfg) {
    if (cfg != nullptr) { m_db_->Set(cfg); }
    for (auto const &item : m_observers_) { item.second->Push(m_db_, item.first); }
}

void Configurable::Pop(std::shared_ptr<DataEntry> const &cfg) {
    for (auto const &item : m_observers_) { item.second->Pop(m_db_, item.first); }
    if (cfg != nullptr) { cfg->Set(m_db_); }
}
void Configurable::Pop(std::shared_ptr<DataEntry> const &cfg) const {
    if (cfg != nullptr) {
        cfg->Set(m_db_);
        for (auto const &item : m_observers_) { item.second->Pop(cfg, item.first); }
    }
}

void Configurable::SetDB(std::shared_ptr<DataEntry> const &cfg) {
    m_db_ = cfg;
    if (m_db_ != nullptr) {
        Push();
        Pop();
    }
}

void Configurable::SetDB(std::shared_ptr<const DataEntry> const &cfg) {
    if (cfg != nullptr) {
        if (m_db_ == nullptr) { m_db_ = data::DataEntry::New(data::DataEntry::DN_TABLE); }
        m_db_->Set(cfg);
        Push();
        Pop();
    }
}
void Configurable::Link(Configurable *other) {
    if (other != nullptr) {
        m_db_->Set(other->m_db_);
        Push();
        Pop();
        other->m_db_ = m_db_;
        other->Pop();
        other->Push();
    }
}
void Configurable::Link(Configurable const *other) {
    if (other != nullptr) {
        m_db_->Set(other->m_db_);
        Push();
        Pop();
    }
}
}  // namespace geometry
}  // namespace simpla