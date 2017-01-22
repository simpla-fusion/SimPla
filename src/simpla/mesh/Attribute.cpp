//
// Created by salmon on 16-10-20.
//

#include "Attribute.h"
#include <simpla/mesh/Patch.h>
#include <simpla/toolbox/Log.h>
#include <typeindex>
#include "Atlas.h"
#include "DataBlock.h"
#include "Mesh.h"
#include "MeshBlock.h"
#include "Worker.h"

namespace simpla {
namespace mesh {

Attribute::Attribute(Mesh *m, const std::shared_ptr<AttributeDesc> &desc, const std::shared_ptr<DataBlock> &d)
    : m_observered_(nullptr), m_mesh_(m), m_desc_(desc), m_data_(d) {
    ASSERT(m_mesh_ != nullptr);
};
Attribute::Attribute(Worker *w, const std::shared_ptr<AttributeDesc> &desc, const std::shared_ptr<DataBlock> &d)
    : m_observered_(w), m_mesh_(nullptr), m_desc_(desc), m_data_(d) {
    if (m_observered_ != nullptr) m_observered_->Connect(this);
};
Attribute::~Attribute() {
    if (m_observered_ != nullptr) m_observered_->Disconnect(this);
}
void Attribute::mesh(Mesh const *m) {
    Finalize();
    m_mesh_ = m;
}

void Attribute::data_block(std::shared_ptr<DataBlock> const &d) {
    Finalize();
    m_data_ = d;
}

void Attribute::SetUp(Mesh const *m, std::shared_ptr<DataBlock> const &d) {
    Finalize();
    m_mesh_ = m;
    m_data_ = d;
}
/**
 *
 * @startuml
 * (*)--> "START"
 *    if "isInitialized ()" then
 *      --> [true] (*)
 *     else
 *       --> [false] Object::Initialize()
 *       if "m_mesh_==nullptr" then
 *         -right->[true]   throw Error
 *         --> (*)
 *       else
 *         --> [false]  if "m_data_==nullptr" then
 *                         --> [true] m_data_= create_data_block();
 *                      endif
 *       endif
 *       --> (*)
 *     endif
 * @enduml
 */
void Attribute::Initialize() {
    if (isInitialized()) { return; }
    Object::Initialize();
    ASSERT(m_mesh_ != nullptr);
    if (m_data_ == nullptr) { m_data_ = create_data_block(); }
    ASSERT(m_data_ != nullptr && !m_data_->empty());
}
void Attribute::Finalize() {
    if (!isInitialized()) { return; }
    Object::Finalize();
}
void Attribute::PreProcess() {
    if (isPrepared()) { return; }
    Object::PreProcess();
    // do sth. at here
}

void Attribute::PostProcess() {
    if (!isPrepared()) { return; }
    // do sth. at here
    m_data_.reset();
    Object::PostProcess();
}

void Attribute::Clear() {
    PreProcess();
    ASSERT(m_data_ != nullptr);
    m_data_->Clear();
}

std::ostream &AttributeDict::Print(std::ostream &os, int indent) const {
    for (auto const &item : m_map_) {
        os << std::setw(indent + 1) << " " << item.second->name() << " = {" << item.second << "}," << std::endl;
    }
    return os;
};

std::pair<std::shared_ptr<AttributeDesc>, bool> AttributeDict::register_attr(
    std::shared_ptr<AttributeDesc> const &desc) {
    auto res = m_map_.emplace(desc->id(), desc);
    return std::make_pair(res.first->second, res.second);
}

void AttributeDict::erase(id_type const &id) {
    auto it = m_map_.find(id);
    if (it != m_map_.end()) {
        m_key_id_.erase(it->second->name());
        m_map_.erase(it);
    }
}

void AttributeDict::erase(std::string const &id) {
    auto it = m_key_id_.find(id);
    if (it != m_key_id_.end()) { erase(it->second); }
}

std::shared_ptr<AttributeDesc> AttributeDict::find(id_type const &id) {
    auto it = m_map_.find(id);
    if (it != m_map_.end()) {
        return it->second;
    } else {
        return nullptr;
    }
};

std::shared_ptr<AttributeDesc> AttributeDict::find(std::string const &id) {
    auto it = m_key_id_.find(id);
    if (it != m_key_id_.end()) {
        return find(it->second);
    } else {
        return nullptr;
    }
};

std::shared_ptr<AttributeDesc> const &AttributeDict::get(std::string const &k) const {
    return m_map_.at(m_key_id_.at(k));
}

std::shared_ptr<AttributeDesc> const &AttributeDict::get(id_type k) const { return m_map_.at(k); }
}
}  // namespace simpla { namespace mesh
