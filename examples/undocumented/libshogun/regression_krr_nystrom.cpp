/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Fredrik Hallgren
 * Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/KrrNystrom.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void test_krr_nystrom()
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

	/* testing features */
	CDenseFeatures<float64_t>* test_features=
			new CDenseFeatures<float64_t>(test_dat);

	/* training labels */
	CRegressionLabels* labels=new CRegressionLabels(lab);

	/* kernel */
	CGaussianKernel* kernel=new CGaussianKernel(features, features, 10, 0.5);

	/* kernel ridge regression and the nystrom approximation */
	float64_t tau=0.0001;
	CKrrNystrom* nystrom=new CKrrNystrom(tau, 10, kernel, labels);

	nystrom->train();

	CLabels* result = nystrom->apply(test_features);

	SG_UNREF(nystrom);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_krr_nystrom();

	exit_shogun();

	return 0;
}

