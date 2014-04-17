/*
 * pythonTypeMap.h
 *
 *  Created on: 2011-12-25
 *      Author: salmon
 */

#ifndef PYTHONTYPEMAP_H_
#define PYTHONTYPEMAP_H_

#include <map>
#include <string>
#include <vector>
#include <boost/any.hpp>
#include <python2.7/object.h>
#include <python2.7/stringobject.h>
#include <python2.7/floatobject.h>
#include <python2.7/intobject.h>
#include <python2.7/tupleobject.h>
#include <python2.7/listobject.h>
#include <python2.7/dictobject.h>
#include <python2.7/pyerrors.h>

#include "defs.h"
#include "fetl/ArrayObject.h"

template<typename T>
void toCXX(PyObject * input, T* v)
{
	toCXX(input, &v);
}

void toCXX(PyObject * input, std::string & v)
{
	if (PyString_Check(input))
	{
		v = std::string(PyString_AS_STRING(input));
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "not a string!");
		throw("not a string!");
	}
}

void toCXX(PyObject * input, Real & v)
{
	if (PyFloat_Check(input))
	{
		v = static_cast<Real>(PyFloat_AsDouble(input));
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "not a double!");
		throw("not a double!");
	}
}

void toCXX(PyObject * input, SizeType & v)
{
	if (PyInt_Check(input))
	{
		v = static_cast<SizeType>(PyInt_AsLong(input));
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "not a integer!");
		throw("not a integer!");
	}
}

PyObject * fromCXX(std::string const & v)
{
	return PyString_FromString(v.c_str());
}

PyObject * fromCXX(SizeType v)
{
	return PyInt_FromLong(static_cast<long>(v));
}

PyObject * fromCXX(Real v)
{
	return PyFloat_FromDouble(static_cast<double>(v));
}

template<typename T, int N>
void toCXX(PyObject * input, nTuple<T, N> & v)
{
	if (PyTuple_Check(input) && PyTuple_Size(input) >= N)
	{
		for (ssize_t i = 0; i < N; ++i)
		{
			toCXX(PyTuple_GetItem(input, i), v[i]);
		}
	}
	else if (PyList_Check(input) && PyList_Size(input) >= N)
	{
		for (ssize_t i = 0; i < N; ++i)
		{
			toCXX(PyList_GetItem(input, i), v[i]);
		}
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "not a tuple/list or too short!");
		throw("not a tuple/list  or too short!");
	}
}

template<typename T, int N>
PyObject * fromCXX(nTuple<T, N> const & v)
{
	PyObject * result = PyTuple_New(N);

	for (int i = 0; i < N; ++i)
	{
		PyTuple_SetItem(result, i, fromCXX(v[i]));
	}
	return result;
}

void toCXX(PyObject * input, boost::any & v)
{
	if (PyInt_Check( input))
	{
		v = PyInt_AsLong(input);
	}
	else if (PyFloat_Check(input))
	{
		v = PyFloat_AsDouble(input);
	}
	else if (PyString_Check(input))
	{
		v = std::string(PyString_AS_STRING(input));
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "unknown type of boost::any !");
		throw(-1);
	}
}

PyObject * fromCXX(boost::any const& v)
{
	PyObject * result = NULL;
	if (v.type() == typeid(long) || v.type() == typeid(SizeType))
	{
		result = PyInt_FromLong(boost::any_cast<long>(v));
	}
	else if (v.type() == typeid(double) || v.type() == typeid(Real))
	{
		result = PyFloat_FromDouble(boost::any_cast<double>(v));
	}
	else if (v.type() == typeid(std::string))
	{
		result = PyString_FromString(boost::any_cast<std::string>(v).c_str());
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "unknown type of boost::any !");
		throw(-1);
	}
	return result;
}
template<typename TKEY, typename TVALUE>
void toCXX(PyObject * input, std::map<TKEY, TVALUE> & v)
{
	if (PyDict_Check(input))
	{
		PyObject *key, *value;
		Py_ssize_t pos = 0;

		TKEY ckey;
		TVALUE cvalue;
		while (PyDict_Next(input, &pos, &key, &value))
		{
			toCXX(key, ckey);
			toCXX(value, cvalue);
			v.insert(typename std::map<TKEY, TVALUE>::value_type(ckey, cvalue));
		}
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "not a Dict!");
		throw(-1);
	}
}

template<typename TKEY, typename TVALUE>
PyObject * fromCXX(std::map<TKEY, TVALUE> const& v)
{
	PyObject * result = PyDict_New();
	for (typename std::map<TKEY, TVALUE>::const_iterator it = v.begin();
			it != v.end(); ++it)
	{
		PyDict_SetItem(result, fromCXX(it->first), fromCXX(it->second));
	}
	return result;
}
#include "fetl/ArrayObject.h"
#include "python2.7/numpy/arrayobject.h"

void toCXX(PyObject * input, ArrayObject & v)
{
	if (PyArray_Check(input))
	{
		int nd = PyArray_NDIM(input);
		npy_intp * pdims = PyArray_DIMS(input);
		npy_intp * pstrides = PyArray_STRIDES(input);
		std::vector<SizeType> dims(pdims, pdims + nd);
		std::vector<SizeType> strides(pdims, pdims + nd);

		ArrayObject(PyArray_DATA(input), dims, strides, typeid(double),
				ArrayObject::CORDER).swap(v);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "not a numpy.ndArray!");
		throw(-1);
	}
}

PyObject * fromCXX(ArrayObject & v)
{
	int nd = v.getNd();
	npy_intp dims[nd];
	npy_intp strides[nd];
	int type = NPY_DOUBLE;
	std::copy(v.getDimensions().begin(), v.getDimensions().end(), dims);
	std::copy(v.getStrides().begin(), v.getStrides().end(), strides);

	if (v.checkType<double>())
	{
		type = NPY_DOUBLE;
	}
	else if (v.checkType<long>())
	{
		type = NPY_LONG;
	}
	else if (v.checkType<unsigned long>())
	{
		type = NPY_ULONG;
	}
	else if (v.checkType<std::complex<double> >())
	{
		type = NPY_CDOUBLE;
	}
	// TODO need improv to more complex array
	return PyArray_SimpleNewFromData(nd,dims,type,reinterpret_cast<void*>(v.getData()));
}
void toCXX(PyObject * input, TR1::shared_ptr<ArrayObject> v)
{
	toCXX(input, *v);
}

PyObject * fromCXX(TR1::shared_ptr<ArrayObject> v)
{
	return fromCXX(*v);
}

void toCXX(PyObject * input,
		SwigValueWrapper<TR1::shared_ptr<ArrayObject> >& v)
{
	toCXX(input, *(&v));
}
PyObject * fromCXX(SwigValueWrapper<TR1::shared_ptr<ArrayObject> >& v)
{
	return fromCXX(*(&v));
}
#endif /* PYTHONTYPEMAP_H_ */
