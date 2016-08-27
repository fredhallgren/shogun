
/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2016 Fredrik Hallgren
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <limits>
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

	m_num_rkhs_basis=m;

	int32_t n=kernel->get_num_vec_lhs();

	REQUIRE(m_num_rkhs_basis <= n, "Number of sampled rows (%d) must be \
less than number of data points (%d)\n", m_num_rkhs_basis, n);
}

void CKRRNystrom::init()
{
	m_num_rkhs_basis=100;
}

SGVector<int32_t> CKRRNystrom::subsample_indices()
{
	int32_t n=kernel->get_num_vec_lhs();
	SGVector<int32_t> temp(n);
	temp.range_fill();
	CMath::permute(temp);
	SGVector<int32_t> col(m_num_rkhs_basis);
	for (index_t i=0; i<m_num_rkhs_basis; ++i)
		col[i]=temp[i];
	CMath::qsort(col.vector, m_num_rkhs_basis);

	return col;
}

bool CKRRNystrom::solve_krr_system()
{

	solve_krr_system_incremental(); // just for testing...


	int32_t n=kernel->get_num_vec_lhs();

	SGVector<float64_t> y=((CRegressionLabels*)m_labels)->get_labels();
	if (y==NULL)
		SG_ERROR("Labels not set.\n");
	SGVector<int32_t> col=subsample_indices();
	SGMatrix<float64_t> K_mm(m_num_rkhs_basis, m_num_rkhs_basis);
	SGMatrix<float64_t> K_nm(n, m_num_rkhs_basis);
	#pragma omp parallel for
	for (index_t j=0; j<m_num_rkhs_basis; ++j)
	{
		for (index_t i=0; i<n; ++i)
			K_nm(i,j)=kernel->kernel(i,col[j]);
	}
	#pragma omp parallel for
	for (index_t i=0; i<m_num_rkhs_basis; ++i)
		memcpy(K_mm.matrix+i*m_num_rkhs_basis, K_nm.get_row_vector(col[i]), m_num_rkhs_basis*sizeof(float64_t));

	Map<MatrixXd> K_mm_eig(K_mm.matrix, m_num_rkhs_basis, m_num_rkhs_basis);
	Map<MatrixXd> K_nm_eig(K_nm.matrix, n, m_num_rkhs_basis);
	MatrixXd K_mn_eig = K_nm_eig.transpose();
	Map<VectorXd> y_eig(y.vector, n);
	VectorXd alphas_eig(m_num_rkhs_basis);

	/* Calculate the Moore-Penrose pseudoinverse */
	MatrixXd Kplus=K_mn_eig*K_nm_eig+m_tau*K_mm_eig;
	SelfAdjointEigenSolver<MatrixXd> solver(Kplus);
	if (solver.info()!=Success)
	{
		SG_WARNING("Eigendecomposition failed.\n")
		return false;
	}

	/* Solve the system for alphas */
	MatrixXd D=solver.eigenvalues().asDiagonal();
	MatrixXd eigvec=solver.eigenvectors();
	float64_t dbl_epsilon=std::numeric_limits<float64_t>::epsilon();
	const float64_t tolerance=m_num_rkhs_basis*dbl_epsilon*D.maxCoeff();
	for (index_t i=0; i<m_num_rkhs_basis; ++i)
	{
		if (D(i,i)<tolerance)
			D(i,i)=0;
		else
			D(i,i)=1/D(i,i);
	}
	MatrixXd pseudoinv=eigvec*D*eigvec.transpose();
	alphas_eig=pseudoinv*K_mn_eig*y_eig;

	/* Expand alpha with zeros to size n */
	SGVector<float64_t> alpha_n(n);
	alpha_n.zero();
	for (index_t i=0; i<m_num_rkhs_basis; ++i)
		alpha_n[col[i]]=alphas_eig[i];
	m_alpha=alpha_n;

	return true;
}



bool CKRRNystrom::solve_krr_system_incremental()
{
	// start just implementing this function to test it... then split the
	// updates to separate function
	// Need to call this anyway ... idea is to do it from the first basis vector

	int32_t n=kernel->get_num_vec_lhs();
	SGVector<float64_t> y_sg=((CRegressionLabels*)m_labels)->get_labels();
	if (y_sg==NULL)
		SG_ERROR("Labels not set.\n");
	SGVector<int32_t> col=subsample_indices(); // m set already

	// Compute kernel matrix Knm - need this but might later be done in incremental method
	SGMatrix<float64_t> K_nm_sg(m_num_rkhs_basis, n);
	SGMatrix<float64_t> K_mm_sg(m_num_rkhs_basis, m_num_rkhs_basis);
	#pragma omp parallel for
	for (index_t j=0; j<m_num_rkhs_basis; ++j)
	{
		for (index_t i=0; i<n; ++i)
			K_nm_sg(i,j)=kernel->kernel(i,col[j]);
	}
	#pragma omp parallel for
	for (index_t i=0; i<m_num_rkhs_basis; ++i)
		memcpy(K_mm_sg.matrix+i*m_num_rkhs_basis, K_nm_sg.get_row_vector(col[i]), m_num_rkhs_basis*sizeof(float64_t));

	Map<MatrixXd> K_nm(K_nm_sg.matrix, n, m_num_rkhs_basis);
	Map<MatrixXd> K_mm(K_mm_sg.matrix, m_num_rkhs_basis, m_num_rkhs_basis);
	Map<VectorXd> y(y_sg.vector, n);
	float64_t gamma=K_nm.col(0).dot(K_nm.col(0))+m_tau*K_mm(0,0);
	//MatrixXd R_i=gamma_1.array().sqrt(); // auto cast to matrix??
	MatrixXd M(m_num_rkhs_basis, m_num_rkhs_basis); // Matrix to update
	M.fill(0);
	M(0,0) = gamma;
	LLT<MatrixXd> llt(M); // calculate initial cholesky
	if (llt.info()!=Eigen::Success)
	{
		SG_WARNING("Cholesky decomposition failed\n"); // should get here if no work...
		return false;
	}
	for (index_t i=2; i<m_num_rkhs_basis; ++i)
	{
		// VectorXd are column vectors for the purposes of lilnear algebra operations
		VectorXd a_i=K_nm.col(i); // or create once and update dymagically?
		VectorXd b_i=K_mm.col(i).head(i);
		MatrixXd A_i=K_nm.leftCols(i-1);
		VectorXd c_i=A_i.transpose()*a_i+m_tau*b_i;
		gamma=a_i.dot(a_i)+m_tau*K_mm(i,i);
		float64_t g_i=sqrt(1+gamma);
		VectorXd u_i;
		u_i << c_i/(1+g_i), g_i, VectorXd::Zero(m_num_rkhs_basis-i);
		VectorXd v_i;
		v_i << c_i/(1+g_i), -1, VectorXd::Zero(m_num_rkhs_basis-i);

		llt.rankUpdate(u_i, 1);
		llt.rankUpdate(v_i, -1);

		VectorXd b=A_i.transpose()*y;
		// TODO: now need to solve ... tricky with zeros everywhere...
		// ... can solve anyway somehow? ... we can just implement our own
		// back/forwards substitution, or call our own, so easy - need to check
		// first though that it worked to calcualte the cholesky

	}

	return true;
}

