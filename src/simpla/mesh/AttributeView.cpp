//
// Created by salmon on 16-10-20.
//

#include "AttributeView.h"
#include <simpla/mesh/Patch.h>
#include <simpla/toolbox/Log.h>
#include <typeindex>
#include "Atlas.h"
#include "Attribute.h"
#include "DataBlock.h"
#include "Mesh.h"
#include "MeshBlock.h"
#include "Worker.h"
namespace simpla {
namespace mesh {

AttributeView::AttributeView(Mesh *m, const std::shared_ptr<Attribute> &desc, const std::shared_ptr<DataBlock> &d)
    : m_observed_(nullptr), m_mesh_(m), m_desc_(desc), m_data_(d) {
    ASSERT(m_mesh_ != nullptr);
};
AttributeView::AttributeView(Worker *w, const std::shared_ptr<Attribute> &desc, const std::shared_ptr<DataBlock> &d)
    : m_observed_(w), m_mesh_(nullptr), m_desc_(desc), m_data_(d) {
    if (m_observed_ != nullptr) m_observed_->Connect(this);
};
AttributeView::~AttributeView() {
    if (m_observed_ != nullptr) m_observed_->Disconnect(this);
}
void AttributeView::mesh(Mesh const *m) {
    Finalize();
    m_mesh_ = m;
}

void AttributeView::data_block(std::shared_ptr<DataBlock> const &d) {
    Finalize();
    m_data_ = d;
}

void AttributeView::SetUp(Mesh const *m, std::shared_ptr<DataBlock> const &d) {
    Finalize();
    m_mesh_ = m;
    m_data_ = d;
}
/**
 *
 * @startuml
 * title     Attribute::Initialize()
 * (*)-->  if "isInitialized ()" then
 *      --> [true] (*)
 *     else
 *       --> [false] Object::Initialize()
 *       if "m_mesh_==nullptr" then
 *         -right->[true]   throw Error
 *         --> (*)
 *       else
 *         --> [false]  if "m_data_==nullptr" then
 *                         --> [true] m_data_= CreateDataBlock();
 *                      endif
 *       endif
 *       --> (*)
 *     endif
 * @enduml
 */
void AttributeView::Initialize() {
    if (isInitialized()) { return; }
    Object::Initialize();
    ASSERT(m_mesh_ != nullptr);
    if (m_data_ == nullptr) { m_data_ = CreateDataBlock(); }
    ASSERT(m_data_ != nullptr && !m_data_->empty());
}
void AttributeView::Finalize() {
    if (!isInitialized()) { return; }
    m_mesh_ = nullptr;
    m_data_.reset();
    Object::Finalize();
}
void AttributeView::PreProcess() {
    if (isPrepared()) { return; }
    Object::PreProcess();
    // do sth. at here
}

void AttributeView::PostProcess() {
    if (!isPrepared()) { return; }
    // do sth. at here
    Object::PostProcess();
}

void AttributeView::Clear() {
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

std::pair<std::shared_ptr<Attribute>, bool> AttributeDict::register_attr(std::shared_ptr<Attribute> const &desc) {
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

std::shared_ptr<Attribute> AttributeDict::find(id_type const &id) {
    auto it = m_map_.find(id);
    if (it != m_map_.end()) {
        return it->second;
    } else {
        return nullptr;
    }
};

std::shared_ptr<Attribute> AttributeDict::find(std::string const &id) {
    auto it = m_key_id_.find(id);
    if (it != m_key_id_.end()) {
        return find(it->second);
    } else {
        return nullptr;
    }
};

std::shared_ptr<Attribute> const &AttributeDict::get(std::string const &k) const { return m_map_.at(m_key_id_.at(k)); }

std::shared_ptr<Attribute> const &AttributeDict::get(id_type k) const { return m_map_.at(k); }
}
}  // namespace simpla { namespace mesh
