/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Fredrik Hallgren
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KRRNYSTROM_H__
#define _KRRNYSTROM_H__

#include <shogun/regression/KernelRidgeRegression.h>

namespace shogun {

class CKRRNystrom : public CKernelRidgeRegression
{
public:
	MACHINE_PROBLEM_TYPE(PT_REGRESSION);

	// Default constructor
	CKRRNystrom();

	// Constructor
	CKRRNystrom(float64_t tau, int32_t m, CKernel* k, CLabels* lab);

	// Default destructor
	virtual ~CKRRNystrom() {}

	// Setters
	inline void set_m(int32_t m) { m_m = m; };

	inline void set_tau(float64_t tau) { m_tau = tau; };

	// Return object name
	virtual const char* get_name() const { return "KRRNystrom"; }

	EMachineType get_classifier_type()
	{
		return CT_KERNELRIDGEREGRESSION;
	}

protected:
	bool solve_krr_system();

private:
	void init();

	// Number of columns/rows to be sampled
	int32_t m_m;

};

}

#endif // _KRRNYSTROM_H__
