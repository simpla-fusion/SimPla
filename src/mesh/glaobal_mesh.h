/*
 * glaobal_mesh.h
 *
 *  Created on: 2014年5月9日
 *      Author: salmon
 */

#ifndef GLAOBAL_MESH_H_
#define GLAOBAL_MESH_H_

namespace simpla
{
template<typename TM>
class GlobalMesh
{
public:

	typedef TM mesh_type;

	GlobalMesh()
	{

	}
	~GlobalMesh()
	{

	}
	template<typename TDict>
	GlobalMesh(TDict const & dict)
	{
	}
	template<typename TDict>
	void Load(TDict const & dict)
	{
	}

	void SetLocal(mesh_type *)
	{
	}

	void Sync(unsigned long)
	{

	}

};
}  // namespace simpla
#endif /* GLAOBAL_MESH_H_ */
