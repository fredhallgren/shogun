
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

#ifndef _KRRNYSTROM_H__
#define _KRRNYSTROM_H__

#include <shogun/regression/KernelRidgeRegression.h>

namespace shogun {

/** @brief Class KRRNystrom implements the Nyström method for kernel ridge
 * regression, using a low-rank approximation to the kernel matrix.
 *
 * The method is equivalent to ordinary kernel ridge regression, but through a
 * low-rank approximation to the original kernel matrix, the full kernel matrix
 * does not need to be computed, and the resulting system of equations for the
 * alphas is cheaper to solve.
 *
 * Given a data matrix X with n training examples along the rows, one chooses
 * a subset m of the training example and projects all training examples upon
 * these. This yields the following approximation to the kernel matrix
 *
 * \f[
 * K \approx K_{n,m}K^+_{m,m}K_{m,n}
 * ]
 * where \f$K_{n,m}\f$ is a submatrix containing all n rows and the m columns
 * corresponding to the m chosen training examples, \f$K_{m,n}\f$ is its
 * transpose and \f$K_{m,m} is the submatrix with the m rows and columns
 * corresponding to the training examples chosen. \f$+\f$ indicates the
 * Moore-Penrose pseudoinverse.
 *
 * The resulting system of equations for the alphas can then be solved through
 * the Woodbury formula, which gives the following equation for the alphas
 *
 * \f[
 * {\bf \alpha} = \frac{1}{\tau} ({\bf y} - K_{n,m} (\tau K_{m,m} + K_{m,n}K_{n,m})^+K_{m,n} {\bf y}
 * ]
 */
class CKRRNystrom : public CKernelRidgeRegression
{
public:
	MACHINE_PROBLEM_TYPE(PT_REGRESSION);

	/** Default constructor */
	CKRRNystrom();

	/** Constructor
	 *
	 * @param tau regularization parameter tau
	 * @param m number of rows/columns to choose
	 * @param k kernel
	 * @param lab labels
	 */
	CKRRNystrom(float64_t tau, int32_t m, CKernel* k, CLabels* lab);

	/** Default destructor */
	virtual ~CKRRNystrom() {}

	/** Set the number of columns/rows to choose
	 *
	 * @param m new m
	 */
	inline void set_m(int32_t m) { m_m = m; };

	/** Set the regularization parameter
	 *
	 * @param tau new tau
	 */
	inline void set_tau(float64_t tau) { m_tau = tau; };

	/** @return object name */
	virtual const char* get_name() const { return "KRRNystrom"; }

	/**
	 * Get the classifier type
	 *
	 * @return classifier type KernelRidgeRegression
	 */
	EMachineType get_classifier_type()
	{
		return CT_KERNELRIDGEREGRESSION;
	}

protected:
	/** Train regression using the Nyström method.
	 *
	 * @return boolean to indicate success
	 */
	bool solve_krr_system();

	/** Sample indices to pick rows/columns from kernel matrix
	 *
	 * @return SGVector<int32_t> with sampled indices
	 */
	SGVector<int32_t> sample_index();
private:
	void init();

	/** Number of columns/rows to be sampled */
	int32_t m_m;

};

}

#endif // _KRRNYSTROM_H__
