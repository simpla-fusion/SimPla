/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "simpla/SIMPLA_config.h"

#include "simpla/geometry/Chart.h"

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
    std::map<id_type, Patch> m_patches_;
    std::multimap<id_type, id_type> m_adjacent_;
    std::multimap<id_type, id_type> m_refine_;
    std::multimap<id_type, id_type> m_coarsen_;

    size_type m_level_ = 0;
    size_type m_max_level_ = 3;
    std::set<Patch> m_layers_[MAX_NUM_OF_LEVEL];

    size_tuple m_refine_ratio_[MAX_NUM_OF_LEVEL] = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    size_tuple m_smallest_dimensions_{8, 8, 8};
    size_tuple m_largest_dimensions_{64, 64, 64};
    index_box_type m_index_box_{{0, 0, 0}, {32, 32, 32}};

    box_type m_box_{{0, 0, 0}, {32, 32, 32}};
    index_tuple m_periodic_dimension_{1, 1, 1};
    point_type m_coarsest_cell_width_ = {1, 1, 1};

    std::shared_ptr<geometry::Chart> m_chart_ = nullptr;
};

Atlas::Atlas(std::string const &s_name) : SPObject(s_name), m_pimpl_(new pimpl_s){};
Atlas::~Atlas(){};
void Atlas::DoUpdate() { SPObject::DoUpdate(); };

std::shared_ptr<data::DataTable> Atlas::Serialize() const {
    auto res = std::make_shared<data::DataTable>();

    res->SetValue("PeriodicDimension", GetPeriodicDimension());
    res->SetValue("IndexOrigin", std::get<0>(GetIndexBox()));
    res->SetValue("Dimensions", GetDimensions());
    res->SetValue("lo", std::get<0>(GetBox()));
    res->SetValue("hi", std::get<1>(GetBox()));

    return (res);
};
void Atlas::Deserialize(const std::shared_ptr<data::DataTable> &cfg) {
    if (cfg == nullptr) { return; }

    SetBox(box_type{cfg->GetValue<point_type>("lo", point_type{0, 0, 0}),
                    cfg->GetValue<point_type>("hi", point_type{1, 1, 1})});
    index_box_type idx_box;
    std::get<0>(idx_box) = cfg->GetValue<nTuple<int, 3>>("IndexOrigin", nTuple<int, 3>{0, 0, 0});
    std::get<1>(idx_box) = cfg->GetValue<nTuple<int, 3>>("Dimensions", nTuple<int, 3>{1, 1, 1});
    SetIndexBox(idx_box);
    index_tuple periodic_dim{0, 0, 0};
    periodic_dim = cfg->GetValue<nTuple<int, 3>>("IndexOrigin", nTuple<int, 3>{0, 0, 0});
    SetPeriodicDimension(periodic_dim);

    Click();
};

void Atlas::Decompose(size_tuple const &d, int local_id) { Click(); };
index_box_type Atlas::FitIndexBox(box_type const &b, int level, int flag) const { return index_box_type{}; }

size_type Atlas::DeletePatch(id_type id) { return m_pimpl_->m_patches_.erase(id); }

id_type Atlas::Push(Patch &&p) {
    auto id = p.GetId();
    m_pimpl_->m_patches_[id].swap(p);
    //    auto res = m_pack_->m_patches_.emplace(p.GetId(), p);
    //    if (!res.second) { p.swap(res.first->second); }
    return id;
}

Patch Atlas::Pop(id_type id) {
    Patch res{id};

    auto it = m_pimpl_->m_patches_.find(id);
    if (it != m_pimpl_->m_patches_.end()) {
        res.swap(it->second);
        m_pimpl_->m_patches_.erase(it);
    }
    return (res);
}
//    auto res = m_pack_->m_patches_.emplace(id, Patch{});
//    if (res.first->second.empty()) { res.first->second = Patch(id); }

size_type Atlas::GetNumOfLevel() const { return m_pimpl_->m_max_level_; }
size_type Atlas::GetMaxLevel() const { return m_pimpl_->m_max_level_; }
void Atlas::SetMaxLevel(size_type l) {
    m_pimpl_->m_max_level_ = l;
    Click();
}
void Atlas::SetRefineRatio(size_tuple const &v, size_type level) {
    m_pimpl_->m_refine_ratio_[level] = v;
    m_pimpl_->m_level_ = std::max(level, m_pimpl_->m_level_);
    Click();
}
size_tuple Atlas::GetRefineRatio(int l) const { return m_pimpl_->m_refine_ratio_[l]; };
void Atlas::SetLargestDimensions(size_tuple const &d) {
    m_pimpl_->m_largest_dimensions_ = d;
    Click();
};
size_tuple Atlas::GetLargestDimensions() const { return m_pimpl_->m_largest_dimensions_; };
void Atlas::SetSmallestDimensions(size_tuple const &d) {
    m_pimpl_->m_smallest_dimensions_ = d;
    Click();
};
size_tuple Atlas::GetSmallestDimensions() const { return m_pimpl_->m_smallest_dimensions_; };
size_tuple Atlas::GetDimensions() const {
    size_tuple d;
    d = std::get<1>(m_pimpl_->m_index_box_) - std::get<0>(m_pimpl_->m_index_box_);
    return d;
}

void Atlas::SetPeriodicDimension(index_tuple const &b) { m_pimpl_->m_periodic_dimension_ = b; }
index_tuple const &Atlas::GetPeriodicDimension() { return m_pimpl_->m_periodic_dimension_; }
void Atlas::SetIndexBox(index_box_type const &b) { m_pimpl_->m_index_box_ = b; };
void Atlas::SetBox(box_type const &b) const { m_pimpl_->m_box_ = b; }
index_box_type Atlas::GetIndexBox() const { return m_pimpl_->m_index_box_; }
box_type Atlas::GetBox() const { return m_pimpl_->m_box_; }
index_tuple Atlas::GetPeriodicDimension() const { return m_pimpl_->m_periodic_dimension_; }

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