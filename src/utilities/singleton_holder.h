/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * SingletonHolder.h
 *
 *  Created on: 2008-04-16
 *      Author: salmon
 */
#ifndef INCLUDE_SINGLETON_HOLDER_H_
#define INCLUDE_SINGLETON_HOLDER_H_
namespace simpla
{

/*
 *@NOTE  Meyers Singleton，
 * Ref:Andrei Alexandrescu,《Ｃ++设计新思维》 候捷 译 p133. Charpt 6.4
 * (  Modern C++ Design --Generic Programming and Design Patterns Applied
 * 2001 Addison Wesley ),
 */
template<class T>
class SingletonHolder
{
public:
	static T & instance()
	{
		if (!pInstance_)
		{
//#pragma omp critical
			//TOD add some for mt critical
			if (!pInstance_)
			{
				static T tmp;
				pInstance_ = &tmp;
			}
		}
		return *pInstance_;
	}
protected:
	SingletonHolder()
	{
	}
	~SingletonHolder()
	{
	}
	static T * volatile pInstance_;
};
template<class T>
T * volatile SingletonHolder<T>::pInstance_ = 0;
}  // namespace simpla
#endif  // INCLUDE_SINGLETON_HOLDER_H_
