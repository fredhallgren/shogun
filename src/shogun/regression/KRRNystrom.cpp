
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Fredrik Hallgren
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/regression/KRRNystrom.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;
using namespace Eigen;

CKRRNystrom::CKRRNystrom() : CKernelRidgeRegression()
{
	init();
}

CKRRNystrom::CKRRNystrom(float64_t tau, int32_t m, CKernel* k, CLabels* lab)
: CKernelRidgeRegression(tau, k, lab)
{
	init();

	m_m=m;
}

void CKRRNystrom::init()
{
	// Initialize parameters
	m_m=1000;  // TODO change to what seems to make sense
	// TODO check that less than n
}

bool CKRRNystrom::solve_krr_system()
{
	SGMatrix<float64_t> kernel_matrix(kernel->get_kernel_matrix());
	int32_t n=kernel_matrix.num_rows;
	SGVector<float64_t> y=((CRegressionLabels*)m_labels)->get_labels();

	// Add tau parameter
	for(index_t i=0; i<n; i++)
		kernel_matrix(i,i)+=m_tau;

	SGVector<int32_t> col(m_m);
	col.random(0,n-m_m+1);
	CMath::qsort(col.vector, m_m);
	SGMatrix<float64_t> K_mm(m_m,m_m);
	SGMatrix<float64_t> K_nm(n,m_m);
	for (index_t i=0; i<m_m*m_m; ++i)
		K_mm[i]=kernel_matrix(col[i/m_m]+i/m_m,col[i%m_m]+i%m_m);
	for (index_t j=0; j<n*m_m; ++j) // TODO Assuming row-first indexing
		K_nm[j]=kernel_matrix(j/m_m,col[j%m_m]+j%m_m);

	// Create Eigen3 objects
	Map<MatrixXd> K_mm_eigen(K_mm.matrix, m_m, m_m);
	Map<MatrixXd> K_nm_eigen(K_nm.matrix, n, m_m);
	Map<VectorXd> y_eigen(y.vector, n);
	Map<VectorXd> alphas_eigen(m_alpha.vector, n);

	// Perform eigendecomposition of K_mm and calculate approximate
	// eigendecomposition (U*D*U^T) of original kernel matrix
	SelfAdjointEigenSolver<MatrixXd> solver(K_mm_eigen);
	// TODO add check for success
	MatrixXd D=solver.eigenvalues().asDiagonal();
	MatrixXd eigvec=solver.eigenvectors(); // TODO confirm normalized and vectors along columns
	MatrixXd U=K_nm_eigen*eigvec*D;

	// Solve system
	alphas_eigen=U*D.inverse()*U.transpose()*y_eigen; // TODO make sure diagonal inverse efficient

	return true;
}
