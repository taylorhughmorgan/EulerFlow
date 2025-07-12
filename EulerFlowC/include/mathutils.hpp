#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include <stdio.h>
#include <vector>
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

#endif