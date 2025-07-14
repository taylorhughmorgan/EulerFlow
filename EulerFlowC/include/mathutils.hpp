#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include <stdio.h>
#include <math.h>

template<typename T>
size_t findClosest(const T arr[], size_t n, const T target)
{
    // find the closest corresponding target value in an array
    size_t left = 0, right = n - 1;
    while (left < right) {
        // if the difference between right and target is greater than left and target, decrement right by one
        if (abs(arr[left] - target) <= abs(arr[right] - target))
            right--;
        else
            left++;
    }
    return left;
}

template<typename T>
void reverse_order(const T arr[], size_t npts, T * arr_out)
{
    // reverse the order of an array
    for (size_t i = 0; i < npts; ++i) {
        arr_out[i] = arr[npts - i - 1];
    }
}
#endif