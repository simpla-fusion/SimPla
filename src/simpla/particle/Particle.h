//
// Created by salmon on 17-6-8.
//

#ifndef SIMPLA_PARTICLE_H
#define SIMPLA_PARTICLE_H

#include <simpla/algebra/Field.h>
#include "simpla/SIMPLA_config.h"

#include "simpla/engine/Attribute.h"

namespace simpla {
namespace engine {
struct MeshBase;
}
class ParticleBase : public engine::Attribute, public data::Serializable {
    SP_OBJECT_HEAD(ParticleBase, engine::Attribute)
   public:
    template <typename... Args>
    explicit ParticleBase(engine::MeshBase* grp, Args&&... args)
        : engine::Attribute(grp, FIBER, 1, typeid(Real), std::forward<Args>(args)...) {
        Initialize();
    };

    ~ParticleBase() override { Finalize(); }

    SP_DEFAULT_CONSTRUCT(ParticleBase);

    void DoInitialize() override;
    void DoFinalize() override;

    void Serialize(data::DataTable& cfg) const override;
    void Deserialize(const data::DataTable& cfg) override;

    void Push(std::shared_ptr<data::DataBlock> const& blk) override;
    std::shared_ptr<data::DataBlock> Pop() override;

    void SetNumberOfAttributes(int n);
    int GetNumberOfAttributes() const;

    void SetNumberOfPIC(size_type n);
    size_type GetNumberOfPIC();

    size_type GetMaxSize() const;

    struct Bucket {
        std::shared_ptr<Bucket> next = nullptr;
        size_type count = 0;
        int* tag = nullptr;
        Real** data = nullptr;
    };

    std::shared_ptr<Bucket> GetBucket(id_type s = NULL_ID);
    std::shared_ptr<Bucket> AddBucket(id_type s, size_type num);
    void RemoveBucket(id_type s);
    std::shared_ptr<Bucket> GetBucket(id_type s = NULL_ID) const;

    void InitialLoad(int const* rnd_type = nullptr, size_type rnd_offset = 0);
    size_type Count(id_type s = NULL_ID) const;
    void Sort();
    void DeepSort();

   private:
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;
};

/** @ingroup physical_object
*  @addtogroup particle particle
*  @{
*	  @brief  @ref particle  is an abstraction from  physical particle or "phase-space sample".
*	  @details
* ## Summary
*  - @ref particle is used to  describe trajectory in  @ref phase_space_7d  .
*  - @ref particle is used to  describe the behavior of  discrete samples of
*    @ref phase_space_7d function  \f$ f\left(t,x,y,z,v_x,v_y,v_z \right) \f$.
*  - @ref particle is a @ref container;
*  - @ref particle is @ref splittable;
*  - @ref particle is a @ref field
* ### Data Structure
*  -  @ref particle is  `unorder_set<Point_s>`
*
* ## Requirements
*- The following table lists the requirements of a particle type  '''P'''
*	Pseudo-Signature    | Semantics
* -------------------- |----------
* ` struct Point_s `   | m_data  type of sample point
* ` P( ) `             | Constructor
* ` ~P( ) `            | Destructor
* ` void  next_time_step(dt, args ...) const; `  | push  m_fluid_sp_ a time interval 'dt'
* ` void  next_time_step(num_of_steps,t0, dt, args ...) const; `  | push  m_fluid_sp_ from time 't0' to 't1' with time
*step 'dt'.
* ` flush_buffer( ) `  | flush input m_buffer to internal m_data container
*
*- @ref particle meets the requirement of @ref container,
* Pseudo-Signature                 | Semantics
* -------------------------------- |----------
* ` push_back(args ...) `          | Constructor
* ` foreach(TFun const & fun)  `   | Destructor
* ` dataset dump() `               | dump/copy 'm_data' into a dataset
*
*- @ref particle meets the requirement of @ref physical_object
*   Pseudo-Signature           | Semantics
* ---------------------------- |----------
* ` print(std::ostream & os) ` | print decription of object
* ` update() `                 | sync internal m_data storage and prepare for execute 'next_time_step'
* ` sync()  `                  | sync. internal m_data with other processes and threads
*
*
* ## Description
* @ref particle   consists of  @ref particle_container and @ref particle_engine .
*   @ref particle_engine  describes the individual behavior of one generate. @ref particle_container
*	  is used to manage these samples.
*
*
* ## Example
*
*  @}
*/

template <typename TM>
class Particle : public ParticleBase {
    typedef Particle<TM> particle_type;
    SP_OBJECT_HEAD(particle_type, engine::Attribute);

   public:
    typedef TM mesh_type;
    static constexpr int iform = FIBER;
    static constexpr int ndims = 3;

   private:
    mesh_type const* m_host_ = nullptr;

   public:
    template <typename... Args>
    Particle(mesh_type* grp, Args&&... args)
        : ParticleBase(grp->GetMesh(), std::forward<Args>(args)...), m_host_(grp) {}

    ~Particle() override = default;

    SP_DEFAULT_CONSTRUCT(Particle);

};  // class Particle
}  // namespace simpla{

#endif  // SIMPLA_PARTICLE_H
