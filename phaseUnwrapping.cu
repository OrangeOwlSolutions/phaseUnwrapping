#define _USE_MATH_DEFINES // for C++
#include <cmath>

#include <thrust\scan.h>
#include <thrust\execution_policy.h>
#include <thrust\transform.h>
#include <thrust\device_ptr.h>

#include "Utilities.cuh"

#define BLOCKSIZE	256

/**************************************/
/* 1D PHASE UNWRAPPING - HOST VERSION */
/**************************************/
template<class T>
void unwrap1D_host(T * __restrict p, const int N) {

	T dp;
	T dps;
	T *dp_corr	= (T *)malloc(N * sizeof(T));
	T *cumsum	= (T *)malloc(N * sizeof(T));
	T cutoff	= M_PI;               /* default value in matlab */

	for (int j = 0; j < N - 1; j++) {

		// --- Incremental phase variation --- dp = diff(p, 1, 1);
		dp = p[j + 1] - p[j];

		// --- Equivalent phase variation in [-pi, pi] --- dps = mod(dp + dp, 2 * pi) - pi;
		dps = (dp + M_PI) - floor((dp + M_PI) / (2 * M_PI)) * (2 * M_PI) - M_PI;

		// --- Preserve variation sign for +pi vs. -pi --- dps(dps == pi & dp > 0, :) = pi;
		if ((dps == -M_PI) && (dp > 0)) dps = M_PI;

		// --- Incremental phase correction --- dp_corr = dps - dp;
		dp_corr[j] = dps - dp;

		// --- Ignore correction when incremental variation is smaller than cutoff --- dp_corr(abs(dp) < cutoff, :) = 0;
		if (fabs(dp) < cutoff) dp_corr[j] = (T)0;

	}

	// --- Find cumulative sum of deltas --- cumsum = cumsum(dp_corr, 1);
	thrust::inclusive_scan(thrust::host, dp_corr, dp_corr + N - 1, cumsum);

	// --- Integrate corrections and add to P to produce smoothed phase values --- p(2 : m, :) = p(2 : m, :) + cumsum(dp_corr, 1);
	thrust::transform(thrust::host, p + 1, p + N, cumsum, p + 1, thrust::plus<T>());

}

template void unwrap1D_host<float> (float  * __restrict, const int);
template void unwrap1D_host<double>(double * __restrict, const int);

/*****************************************/
/* 1D PHASE UNWRAPPING - GLOBAL FUNCTION */
/*****************************************/
template<class T>
__global__ void unwrap_1D_global(T * __restrict__ p, T * __restrict__ dp_corr, T * __restrict__ cumsum, T cutoff, const int N) {

	const int tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= N - 1) return;

	T dp, dps;

	// --- Incremental phase variation --- dp = diff(p, 1, 1);
	dp = p[tid + 1] - p[tid];

	// --- Equivalent phase variation in [-pi, pi] --- dps = mod(dp + dp, 2 * pi) - pi;
	dps = (dp + M_PI) - floor((dp + M_PI) / (2 * M_PI)) * (2 * M_PI) - M_PI;

	// --- Preserve variation sign for +pi vs. -pi --- dps(dps == pi & dp > 0, :) = pi;
	if ((dps == -M_PI) && (dp > 0)) dps = M_PI;

	// --- Incremental phase correction --- dp_corr = dps - dp;
	dp_corr[tid] = dps - dp;

	// --- Ignore correction when incremental variation is smaller than cutoff --- dp_corr(abs(dp) < cutoff, :) = 0;
	if (fabs(dp) < cutoff) dp_corr[tid] = (T)0;

}

/****************************************/
/* 1D PHASE UNWRAPPING - DEVICE VERSION */
/****************************************/
template<class T>
void unwrap1D_device(T * __restrict__ p, const int N) {

	T *dp_corr;	gpuErrchk(cudaMalloc((void**)&dp_corr, N * sizeof(T)));
	T *cumsum;	gpuErrchk(cudaMalloc((void**)&cumsum, N * sizeof(T)));
	T cutoff = M_PI;               /* default value in matlab */

	unwrap_1D_global << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(p, dp_corr, cumsum, cutoff, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// --- Find cumulative sum of deltas --- cumsum = cumsum(dp_corr, 1);
	thrust::inclusive_scan(thrust::device_pointer_cast(dp_corr), thrust::device_pointer_cast(dp_corr) + N - 1, thrust::device_pointer_cast(cumsum));

	// --- Integrate corrections and add to P to produce smoothed phase values --- p(2 : m, :) = p(2 : m, :) + cumsum(dp_corr, 1);
	thrust::transform(thrust::device_pointer_cast(p) + 1, thrust::device_pointer_cast(p) + N, thrust::device_pointer_cast(cumsum), thrust::device_pointer_cast(p) + 1, thrust::plus<T>());

	gpuErrchk(cudaFree(dp_corr));
	gpuErrchk(cudaFree(cumsum));

}

template void unwrap1D_device<float> (float  * __restrict, const int);
template void unwrap1D_device<double>(double * __restrict, const int);
