/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "Atlas.h"
#include "Patch.h"
//#include "TransitionMap.h"
//#include "simpla/utilities/BoxUtility.h"

namespace simpla {
namespace engine {

struct Atlas::pimpl_s {
    static constexpr int MAX_NUM_OF_LEVEL = 5;

    typedef typename std::multimap<id_type, id_type>::iterator link_iterator;
    typedef typename std::multimap<id_type, id_type>::const_iterator const_link_iterator;
    typedef std::pair<const_link_iterator, const_link_iterator> multi_links_type;
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;
    std::multimap<id_type, id_type> m_adjacent_;
    std::multimap<id_type, id_type> m_refine_;
    std::multimap<id_type, id_type> m_coarsen_;

    size_type m_level_ = 0;
    size_type m_max_level_ = 3;
    std::set<std::shared_ptr<Patch>> m_layers_[MAX_NUM_OF_LEVEL];

    size_tuple m_periodic_dimension_ = {1, 1, 1};
    size_tuple m_refine_ratio_[MAX_NUM_OF_LEVEL] = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    size_tuple m_smallest_dimensions_{16, 16, 16};
    size_tuple m_largest_dimensions_{64, 64, 64};

    index_box_type m_index_box_{{0, 0, 0}, {64, 64, 32}};
};

Atlas::Atlas() : m_pimpl_(new pimpl_s){};
Atlas::~Atlas(){};
void Atlas::SetUp(){};
std::shared_ptr<data::DataTable> Atlas::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    res->SetValue("PeriodicDimension", m_pimpl_->m_periodic_dimension_);
    res->SetValue("IndexBox", m_pimpl_->m_index_box_);

    return res;
};
void Atlas::Deserialize(const std::shared_ptr<data::DataTable> &cfg) {
    if (cfg == nullptr) { return; }
    size_tuple period_dimensions{0, 0, 0};
    period_dimensions = cfg->GetValue<nTuple<int, 3>>("PeriodicDimension", nTuple<int, 3>{0, 0, 1});
    SetPeriodicDimension(period_dimensions);
};

void Atlas::Decompose(size_tuple const &d, int local_id){};
index_box_type Atlas::FitIndexBox(box_type const &b, int level, int flag) const { return index_box_type{}; }

// std::shared_ptr<Patch> Atlas::AddBlock(id_type id, std::shared_ptr<Patch> m) {
//    auto res = m_pimpl_->m_layers_[m->GetLevel()].emplace(m);
//    if (m->GetLevel() > m_pimpl_->m_level_) { m_pimpl_->m_level_ = m->GetLevel(); }
//    return *res.first;
//};
// std::shared_ptr<Patch> Atlas::GetBlock(id_type id) const { return m_pimpl_->m_patches_.at(id); };
// size_type Atlas::DeletePatch(id_type id) {
//    auto p = GetBlock(id);
//    if (p != nullptr) { m_pimpl_->m_layers_[p->GetLevel()].erase(p); }
//    return m_pimpl_->m_patches_.erase(id);
//};
size_type Atlas::DeletePatch(id_type id) { return m_pimpl_->m_patches_.erase(id); }

id_type Atlas::PushPatch(std::shared_ptr<Patch> p) {
    if (p != nullptr) {
        auto res = m_pimpl_->m_patches_.emplace(p->GetId(), p);
        if (!res.second) { res.first->second->Merge(*p); }
    }
    return (p == nullptr) ? NULL_ID : p->GetId();
}

std::shared_ptr<Patch> Atlas::PopPatch(id_type id) {
    auto res = m_pimpl_->m_patches_.emplace(id, nullptr);
    if (res.first->second == nullptr) { res.first->second = std::make_shared<Patch>(id); }
    return res.first->second;
}

void Atlas::SetPeriodicDimension(size_tuple const &d) { m_pimpl_->m_periodic_dimension_ = d; }
size_tuple const &Atlas::GetPeriodicDimension() const { return m_pimpl_->m_periodic_dimension_; }

// std::set<std::shared_ptr<Patch>> const &Atlas::Level(int level) const { return m_pimpl_->m_layers_[level]; };

size_type Atlas::GetNumOfLevel() const { return m_pimpl_->m_max_level_; }
size_type Atlas::GetMaxLevel() const { return m_pimpl_->m_max_level_; }
void Atlas::SetMaxLevel(size_type l) { m_pimpl_->m_max_level_ = l; }
void Atlas::SetRefineRatio(size_tuple const &v, size_type level) {
    m_pimpl_->m_refine_ratio_[level] = v;
    m_pimpl_->m_level_ = std::max(level, m_pimpl_->m_level_);
}
size_tuple Atlas::GetRefineRatio(int l) const { return m_pimpl_->m_refine_ratio_[l]; };
void Atlas::SetLargestDimensions(size_tuple const &d) { m_pimpl_->m_largest_dimensions_ = d; };
size_tuple Atlas::GetLargestDimensions() const { return m_pimpl_->m_largest_dimensions_; };
void Atlas::SetSmallestDimensions(size_tuple const &d) { m_pimpl_->m_smallest_dimensions_ = d; };
size_tuple Atlas::GetSmallestDimensions() const { return m_pimpl_->m_smallest_dimensions_; };
size_tuple Atlas::GetDimensions() const {
    size_tuple d;
    d = std::get<1>(m_pimpl_->m_index_box_) - std::get<0>(m_pimpl_->m_index_box_);
    return d;
}

void Atlas::SetIndexBox(index_box_type ibx) { m_pimpl_->m_index_box_ = ibx; }
index_box_type Atlas::GetIndexBox() const { return m_pimpl_->m_index_box_; }

// size_type Atlas::size(int level) const { return m_backend_->m_layer_[level].size(); }
// void Atlas::max_level(int ml) { m_backend_->m_max_level_ = ml; }
// int Atlas::max_level() const { return m_backend_->m_max_level_; }
// bool Atlas::has(id_type id) const { return m_backend_->m_nodes_.find(id) != m_backend_->m_nodes_.end(); };
//
// RectMesh *Atlas::find(id_type id) {
//    auto it = m_backend_->m_nodes_.find(id);
//    if (it != m_backend_->m_nodes_.end()) {
//        return it->second.get();
//    } else {
//        return nullptr;
//    }
//}
//
// RectMesh const *Atlas::find(id_type id) const {
//    auto it = m_backend_->m_nodes_.find(id);
//    if (it != m_backend_->m_nodes_.end()) {
//        return it->second.get();
//    } else {
//        return nullptr;
//    }
//}
//
// RectMesh *Atlas::at(id_type id) { return (m_backend_->m_nodes_.at(id)).get(); };
//
// RectMesh const *Atlas::at(id_type id) const { return (m_backend_->m_nodes_.at(id)).get(); };
//
// RectMesh const *Atlas::Connect(std::shared_ptr<RectMesh> const &p_m, RectMesh const *hint) {
////    m_pimpl_->m_nodes_.emplace(std::make_pair(p_m->id(), p_m));
//    return p_m.get();
//};
//
// std::ostream &Atlas::Print(std::ostream &os, int indent) const {
////    os << std::setw(indent) << "*" << name() << std::endl;
////    for (auto const &item : m_pimpl_->m_nodes_) {
////        os << "|" << std::setw(indent + 5 + item.second->level()) << std::setfill('-') << "> " << std::setfill(' ')
////           << std::setw(10) << std::left << item.first << std::right << " = {";
////        item.second->Print(os, indent + 1);
////        os << "}," << std::endl;
////    }
//    return os;
//};
//
// void Atlas::Load(const data::DataTable &) { UNIMPLEMENTED; }
//
// void Atlas::Save(data::DataTable *) const { UNIMPLEMENTED; }
//
// void Atlas::Sync(id_type id)
//{
//    unlink(id);
//
//    auto it = m_nodes_.find(id);
//
//    if (it == m_nodes_.end())
//    {
//        return;
//    } else if (id != it->second->id())
//    {
//        m_nodes_.Finalizie(it);
//        return;
//    }
//    RectMesh const &m = *(it->second);
//    assert(m.level() < MAX_NUM_OF_LEVEL);
//    m_max_level_ = std::max(m_max_level_, m.level());
//    m_layer_[m.level()].Connect(id);
//
//    for (auto const &item:m_nodes_) { if (item.first != id) { link(id, item.first); }}
//}
//
// void Atlas::link(id_type i0, id_type i1)
//{
//    assert(has(i0) && find(i1));
//    RectMesh const &m0 = *at(i0);
//    RectMesh const &m1 = *at(i1);
//    int l0 = m0.level();
//    int l1 = m1.level();
//    box_type b0 = m0.box();
//    box_type b1 = m1.box();
//    vector_type dx = m0.dx();
//    vector_type L;//= m0.period_length();
//    switch (l0 - l1)
//    {
//        case 0:
//            if (utilities::check_adjoining(b0, b1, dx, L))
//            {
//                m_adjacent_.Connect(i0, i1);
//                m_adjacent_.Connect(i1, i0);
//            }
//            break;
//
//        case -1:
//            if (utilities::check_overlapping(b0, b1))
//            {
//                m_refine_.Connect(i0, i1);
//                m_coarsen_.Connect(i1, i0);
//            }
//            break;
//        case 1:
//            if (utilities::check_overlapping(b0, b1))
//            {
//                m_coarsen_.Connect(i0, i1);
//                m_refine_.Connect(i1, i0);
//            }
//            break;
//        default:
//            break;
//    }
//
//    return l0 - l1;
//}
//
// void Atlas::unlink(id_type id)
//{
//    m_adjacent_.Finalizie(id);
//    m_refine_.Finalizie(id);
//    m_coarsen_.Finalizie(id);
//    for (int i = 0; i < MAX_NUM_OF_LEVEL; ++i) { m_layer_[i].Finalizie(id); }
//}
//
//
// void Atlas::update_all()
//{
//    for (auto ib = m_nodes_.begin(), ie = m_nodes_.end(); ib != ie; ++ib)
//        for (auto it = ib; it != ie; ++it) { link(ib->first, it->first); }
//    m_max_level_ = 0;
//    for (unsigned int i = 0; i < MAX_NUM_OF_LEVEL; ++i)
//    {
//        if (m_layer_[i].size() == 0) { break; }
//        m_max_level_ = i;
//    }
//
//}
//
//
// std::shared_ptr<TransitionMapBase>
// Atlas::add_adjacency(MeshBlockId first, MeshBlockId second)
//{
//    return add_adjacency(get_block(first), get_block(second));
//}
//
// std::shared_ptr<TransitionMapBase>
// Atlas::add_adjacency(std::shared_ptr<const RectMesh> first, std::shared_ptr<const RectMesh> second)
//{
//    return add_adjacency(std::make_shared<TransitionMapBase>(*first, *second));
//}
//
// std::tuple<std::shared_ptr<TransitionMapBase>, std::shared_ptr<TransitionMapBase>>
// Atlas::add_connection(std::shared_ptr<const RectMesh> first, std::shared_ptr<const RectMesh> second)
//{
//    return std::make_tuple(add_adjacency(first, second), add_adjacency(second, first));
//}
}  // namespace engine
}  // namespace simpla{namespace mesh_as{