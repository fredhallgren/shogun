
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
	col.random(0,n-m_m);
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

	/* Calculate the Moore-Penrose pseudoinverse */
	MatrixXd Kplus = m_tau*K_mm_eig + K_mn_eig*K_nm_eig;
	SelfAdjointEigenSolver<MatrixXd> solver(Kplus);
	if (solver.info()!=Success)
	{
		SG_WARNING("Eigendecomposition failed.")
		return false;
	}
	/* Solve the system for alphas */
	MatrixXd D=solver.eigenvalues().asDiagonal();
	MatrixXd eigvec=solver.eigenvectors(); // TODO confirm normalized and vectors along columns
	MatrixXd pseudoinv=eigvec*D.inverse()*eigvec.transpose(); // TODO confirm diagonal inverse efficient
	alphas_eig=1.0/m_tau*(y_eig-K_nm_eig*pseudoinv*K_mn_eig*y_eig); // TODO maybe split up

	return true;
}
