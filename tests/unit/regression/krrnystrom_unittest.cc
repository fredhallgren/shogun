/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2016 Fredrik Hallgren
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
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

#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/KRRNystrom.h>
#include <shogun/regression/KernelRidgeRegression.h>
#include <gtest/gtest.h>

using namespace shogun;


// TODO test separate functions, and test the solving of the system through the
// alphas, take a few steps back

TEST(KRRNystrom, compare_to_KRR)
{
	/* data matrix dimensions */
	index_t num_vectors=30;
	index_t num_features=1;

	/* training label data */
	SGVector<float64_t> lab(num_vectors);

	/* fill data matrix and labels */
	SGMatrix<float64_t> train_dat(num_features, num_vectors);
	SGMatrix<float64_t> test_dat(num_features, num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
	{
		/* labels are linear plus noise */
		lab.vector[i]=i+CMath::normal_random(0, 1.0);
		train_dat.matrix[i] = i;
		test_dat.matrix[i] = i;
	}

	/* training features */
	CDenseFeatures<float64_t>* features=
			new CDenseFeatures<float64_t>(train_dat);
	CDenseFeatures<float64_t>* features_krr=
			new CDenseFeatures<float64_t>(train_dat);

	/* testing features */
	CDenseFeatures<float64_t>* test_features=
			new CDenseFeatures<float64_t>(test_dat);

	/* training labels */
	CRegressionLabels* labels=new CRegressionLabels(lab);
	CRegressionLabels* labels_krr=new CRegressionLabels(lab);

	/* kernel */
	CGaussianKernel* kernel=new CGaussianKernel(features, features, 10, 0.5);
	CGaussianKernel* kernel_krr=new CGaussianKernel(features, features_krr, 10, 0.5);

	/* kernel ridge regression and the nystrom approximation */
	float64_t tau=0.0001;
	CKRRNystrom* nystrom=new CKRRNystrom(tau, 10, kernel, labels);
	CKernelRidgeRegression* krr=new CKernelRidgeRegression(tau, kernel_krr, labels_krr);

	nystrom->train();
	krr->train();

	CRegressionLabels* result = (CRegressionLabels *)nystrom->apply(test_features);
	CRegressionLabels* result_krr = (CRegressionLabels *)krr->apply(test_features);

	//for (index_t i=0; i<num_vectors; ++i) {
	index_t i = 0;
		EXPECT_NEAR(result->get_label(i), result_krr->get_label(i),1E-1);
	//}

	SG_UNREF(nystrom);
	SG_UNREF(krr);

}
