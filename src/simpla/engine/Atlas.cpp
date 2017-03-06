/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "Atlas.h"
#include "TransitionMap.h"

//#include "simpla/toolbox/BoxUtility.h"

namespace simpla {
namespace engine {

struct Atlas::pimpl_s {
    static constexpr int MAX_NUM_OF_LEVEL = 5;
    size_tuple m_refine_ratio_ = {2, 2, 2};
    point_type m_origin_{0, 0, 0};
    box_type m_bound_box_{{0, 0, 0}, {1, 1, 1}};
    std::vector<point_type> m_dx_;

    typedef typename std::multimap<id_type, id_type>::iterator link_iterator;
    typedef typename std::multimap<id_type, id_type>::const_iterator const_link_iterator;
    typedef std::pair<const_link_iterator, const_link_iterator> multi_links_type;
    std::map<id_type, std::shared_ptr<MeshBlock>> m_nodes_;
    std::multimap<id_type, id_type> m_adjacent_;
    std::multimap<id_type, id_type> m_refine_;
    std::multimap<id_type, id_type> m_coarsen_;
    std::set<id_type> m_layer_[MAX_NUM_OF_LEVEL];
    std::vector<std::map<id_type, MeshBlock>> m_levels_;
};

Atlas::Atlas() : m_pimpl_(new pimpl_s){};
Atlas::~Atlas(){};

bool Atlas::Update() { return SPObject::Update(); };
size_type Atlas::GetNumOfLevels() const { return m_pimpl_->m_levels_.size(); };
void Atlas::SetDx(point_type const &p) { m_pimpl_->m_dx_[0] = p; };
point_type const &Atlas::GetDx(int l) { return m_pimpl_->m_dx_[l]; }
void Atlas::SetOrigin(point_type const &p) { m_pimpl_->m_origin_ = p; };
point_type const &Atlas::GetOrigin() const { return m_pimpl_->m_origin_; };
void Atlas::SetBox(box_type const &b) { m_pimpl_->m_bound_box_ = b; };
box_type const &Atlas::GetBox() const { return m_pimpl_->m_bound_box_; };

index_box_type Atlas::FitIndexBox(box_type const &b, int level, int flag) const {}
MeshBlock const &Atlas::AddBlock(box_type const &, int level){};
MeshBlock const &Atlas::AddBlock(index_box_type const &, int level) {}
MeshBlock const &Atlas::AddBlock(MeshBlock const &m) {
    return m_pimpl_->m_levels_[m.GetLevel()].emplace(m.GetGUID(), m).first->second;
}
MeshBlock const &Atlas::GetBlock(id_type id, int level) const { return m_pimpl_->m_levels_[level].at(id); };
size_type Atlas::EraseBlock(id_type id, int level) { return m_pimpl_->m_levels_[level].erase(id); };
size_type Atlas::EraseBlock(MeshBlock const &m) { return EraseBlock(m.GetGUID(), m.GetLevel()); };
MeshBlock const &Atlas::CoarsenBlock(id_type, int level){};
MeshBlock const &Atlas::CoarsenBlock(MeshBlock const &){};
MeshBlock const &Atlas::RefineBlock(id_type, box_type const &, int level){};
MeshBlock const &Atlas::RefineBlock(MeshBlock const &, box_type const &){};
MeshBlock const &Atlas::RefineBlock(id_type, index_box_type const &, int level){};
MeshBlock const &Atlas::RefineBlock(MeshBlock const &, index_box_type const &){};

void Atlas::Accept(std::function<void(MeshBlock const &)> const &fun, int level) const {
    for (auto const &item : m_pimpl_->m_levels_[level]) { fun(item.second); }
};
std::map<id_type, MeshBlock> const &Atlas::GetBlockList(int level) const { return m_pimpl_->m_levels_[level]; };

//
// size_type Atlas::count(int level) const { return m_backend_->m_layer_[level].size(); }
//
// void Atlas::max_level(int ml) { m_backend_->m_max_level_ = ml; }
//
// int Atlas::max_level() const { return m_backend_->m_max_level_; }
//
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
//            if (toolbox::check_adjoining(b0, b1, dx, L))
//            {
//                m_adjacent_.Connect(i0, i1);
//                m_adjacent_.Connect(i1, i0);
//            }
//            break;
//
//        case -1:
//            if (toolbox::check_overlapping(b0, b1))
//            {
//                m_refine_.Connect(i0, i1);
//                m_coarsen_.Connect(i1, i0);
//            }
//            break;
//        case 1:
//            if (toolbox::check_overlapping(b0, b1))
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