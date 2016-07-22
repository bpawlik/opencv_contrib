/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_BM3D_DENOISING_TRANSFORMS_HPP__
#define __OPENCV_BM3D_DENOISING_TRANSFORMS_HPP__

#include "bm3d_denoising_transforms_1D.hpp"
#include "bm3d_denoising_transforms_2D.hpp"

namespace cv
{
namespace xphoto
{

template <typename T>
inline static void shrink(T &val, T &nonZeroCount, const T &threshold)
{
    if (std::abs(val) < threshold)
        val = 0;
    else
        ++nonZeroCount;
}

template <typename T>
inline static void hardThreshold2D(T *dst, T *thrMap, const int &templateWindowSizeSq)
{
    for (int i = 1; i < templateWindowSizeSq; ++i)
    {
        if (std::abs(dst[i] < thrMap[i]))
            dst[i] = 0;
    }
}

template <int N, typename T, typename DT, typename CT>
inline static T HardThreshold(BlockMatch<T, DT, CT> *z, const int &n, T *&thrMap)
{
    T nonZeroCount = 0;

    for (int i = 0; i < N; ++i)
        shrink(z[i][n], nonZeroCount, *thrMap++);

    return nonZeroCount;
}

template <typename T, typename DT, typename CT>
inline static T HardThreshold(BlockMatch<T, DT, CT> *z, const int &n, T *&thrMap, const int &N)
{
    T nonZeroCount = 0;

    for (int i = 0; i < N; ++i)
        shrink(z[i][n], nonZeroCount, *thrMap++);

    return nonZeroCount;
}

template <int N, typename T, typename DT, typename CT>
inline static int WienerFiltering(BlockMatch<T, DT, CT> *zSrc, BlockMatch<T, DT, CT> *zBasic, const int &n, T *&thrMap)
{
    int wienerCoeffs = 0;

    for (int i = 0; i < N; ++i)
    {
        // Possible optimization point here to get rid of floats and casts
        int basicSq = zBasic[i][n] * zBasic[i][n];
        int sigmaSq = *thrMap * *thrMap;
        int denom = basicSq + sigmaSq;
        float wie = (denom == 0) ? 1.0f : ((float)basicSq / (float)denom);

        zBasic[i][n] = (T)(zSrc[i][n] * wie);
        wienerCoeffs += (int)wie;
        ++thrMap;
    }

    return wienerCoeffs;
}

template <typename T, typename DT, typename CT>
inline static int WienerFiltering(BlockMatch<T, DT, CT> *zSrc, BlockMatch<T, DT, CT> *zBasic, const int &n, T *&thrMap, const unsigned &N)
{
    int wienerCoeffs = 0;

    for (unsigned i = 0; i < N; ++i)
    {
        // Possible optimization point here to get rid of floats and casts
        int basicSq = zBasic[i][n] * zBasic[i][n];
        int sigmaSq = *thrMap * *thrMap;
        int denom = basicSq + sigmaSq;
        float wie = (denom == 0) ? 1.0f : ((float)basicSq / (float)denom);

        zBasic[i][n] = (T)(zSrc[i][n] * wie);
        wienerCoeffs += (int)wie;
        ++thrMap;
    }

    return wienerCoeffs;
}

/// 1D and 2D threshold map coefficients. Implementation dependent, thus stored
/// together with transforms.

static void calcHaarCoefficients1D(cv::Mat &coeff1D, const int &numberOfElements)
{
    // Generate base array and initialize with zeros
    cv::Mat baseArr = cv::Mat::zeros(numberOfElements, numberOfElements, CV_32FC1);

    // Calculate base array coefficients.
    int currentRow = 0;
    for (int i = numberOfElements; i > 0; i /= 2)
    {
        for (int k = 0, sign = -1; k < numberOfElements; ++k)
        {
            // Alternate sign every i-th element
            if (k % i == 0)
                sign *= -1;

            // Move to the next row every 2*i-th element
            if (k != 0 && (k % (2 * i) == 0))
                ++currentRow;

            baseArr.at<float>(currentRow, k) = sign * 1.0f / i;
        }
        ++currentRow;
    }

    // Square each elements of the base array
    float *ptr = baseArr.ptr<float>(0);
    for (unsigned i = 0; i < baseArr.total(); ++i)
        ptr[i] = ptr[i] * ptr[i];

    // Multiply baseArray with 1D vector of ones
    cv::Mat unitaryArr = cv::Mat::ones(numberOfElements, 1, CV_32FC1);
    coeff1D = baseArr * unitaryArr;
}

// Method to generate threshold coefficients for 1D transform depending on the number of elements.
static void fillHaarCoefficients1D(float *thrCoeff1D, int &idx, const int &numberOfElements)
{
    cv::Mat coeff1D;
    calcHaarCoefficients1D(coeff1D, numberOfElements);

    // Square root the array to get standard deviation
    float *ptr = coeff1D.ptr<float>(0);
    for (unsigned i = 0; i < coeff1D.total(); ++i)
    {
        ptr[i] = std::sqrt(ptr[i]);
        thrCoeff1D[idx++] = ptr[i];
    }
}

// Method to generate threshold coefficients for 2D transform depending on the number of elements.
static void fillHaarCoefficients2D(float *thrCoeff2D, const int &templateWindowSize)
{
    cv::Mat coeff1D;
    calcHaarCoefficients1D(coeff1D, templateWindowSize);

    // Calculate 2D array
    cv::Mat coeff1Dt;
    cv::transpose(coeff1D, coeff1Dt);
    cv::Mat coeff2D = coeff1D * coeff1Dt;

    // Square root the array to get standard deviation
    float *ptr = coeff2D.ptr<float>(0);
    for (unsigned i = 0; i < coeff2D.total(); ++i)
        thrCoeff2D[i] = std::sqrt(ptr[i]);
}

// Method to calculate 1D threshold map based on the maximum number of elements
// Allocates memory for the output array.
static void calcHaarThresholdMap1D(float *&thrMap1D, const int &numberOfElements)
{
    CV_Assert(numberOfElements > 0);

    // Allocate memory for the array
    const int arrSize = (numberOfElements << 1) - 1;
    if (thrMap1D == NULL)
        thrMap1D = new float[arrSize];

    for (int i = 1, idx = 0; i <= numberOfElements; i *= 2)
        fillHaarCoefficients1D(thrMap1D, idx, i);
}

// Method to calculate 2D threshold map based on the maximum number of elements
// Allocates memory for the output array.
static void calcHaarThresholdMap2D(float *&thrMap2D, const int &templateWindowSize)
{
    // Allocate memory for the array
    if (thrMap2D == NULL)
        thrMap2D = new float[templateWindowSize * templateWindowSize];

    fillHaarCoefficients2D(thrMap2D, templateWindowSize);
}

// Method to calculate 3D threshold map based on the maximum number of elements.
// Allocates memory for the output array.
template <typename T>
static void calcHaarThresholdMap3D(
    T *&outThrMap1D,
    const float &hardThr1D,
    const int &templateWindowSize,
    const int &groupSize)
{
    const int templateWindowSizeSq = templateWindowSize * templateWindowSize;

    // Allocate memory for the output array
    if (outThrMap1D == NULL)
        outThrMap1D = new T[templateWindowSizeSq * ((groupSize << 1) - 1)];

    // Generate 1D coefficients map
    float *thrMap1D = NULL;
    calcHaarThresholdMap1D(thrMap1D, groupSize);

    // Generate 2D coefficients map
    float *thrMap2D = NULL;
    calcHaarThresholdMap2D(thrMap2D, templateWindowSize);

    // Generate 3D threshold map
    T *thrMapPtr1D = outThrMap1D;
    for (int i = 1, ii = 0; i <= groupSize; ++ii, i *= 2)
    {
        float coeff = (i == 1) ? 1.0f : std::sqrt(2.0f * std::log((float)i));
        for (int jj = 0; jj < templateWindowSizeSq; ++jj)
        {
            for (int ii1 = 0; ii1 < (1 << ii); ++ii1)
            {
                int indexIn1D = (1 << ii) - 1 + ii1;
                int indexIn2D = jj;
                int thr = static_cast<int>(thrMap1D[indexIn1D] * thrMap2D[indexIn2D] * hardThr1D * coeff);

                // Set DC component to zero
                if (jj == 0 && ii1 == 0)
                    thr = 0;

                *thrMapPtr1D++ = cv::saturate_cast<T>(thr);
            }
        }
    }

    delete[] thrMap1D;
    delete[] thrMap2D;
}

}  // namespace xphoto
}  // namespace cv

#endif