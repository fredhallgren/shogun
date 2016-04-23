
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
	m_m=1000;
}

bool CKRRNystrom::solve_krr_system()
{
	int32_t n=kernel->get_num_vec_lhs(); // TODO ever different from rhs?

	ASSERT(m_m <= n);

	SGVector<float64_t> y=((CRegressionLabels*)m_labels)->get_labels();

	SGVector<int32_t> col(m_m);
	col.random(0,n-m_m+1);
	CMath::qsort(col.vector, m_m);
	SGMatrix<float64_t> K_mm(m_m,m_m);
	SGMatrix<float64_t> K_nm(n,m_m);
	for (index_t i=0; i<m_m*m_m; ++i)
		K_mm[i]=kernel->kernel(col[i/m_m]+i/m_m,col[i%m_m]+i%m_m);
	for (index_t j=0; j<n*m_m; ++j) // TODO Assuming row-first indexing
		K_nm[j]=kernel->kernel(j/m_m,col[j%m_m]+j%m_m);

	Map<MatrixXd> K_mm_eig(K_mm.matrix, m_m, m_m);
	Map<MatrixXd> K_nm_eig(K_nm.matrix, n, m_m);
	MatrixXd K_mn_eig = K_nm_eig.transpose();
	Map<VectorXd> y_eig(y.vector, n);
	Map<VectorXd> alphas_eig(m_alpha.vector, n);

	MatrixXd Kplus = m_tau*K_mm_eig + K_mn_eig*K_nm_eig;
	SelfAdjointEigenSolver<MatrixXd> solver(Kplus); // TODO add check for success
	MatrixXd D=solver.eigenvalues().asDiagonal();
	MatrixXd eigvec = solver.eigenvectors(); // TODO confirm normalized and vectors along columns
	MatrixXd pseudoinv = eigvec*D.inverse()*eigvec.transpose(); // TODO confirm diagonal inverse efficient
	alphas_eig=1.0/m_tau*(y_eig-K_nm_eig*pseudoinv*K_mn_eig*y_eig); // TODO maybe split up

	return true;
}
