# Feature Detector FAST

This is a highly optimised implementation of the FAST feature detector, described in "Machine Learning for High-Speed Corner Detection" by Rosten and Drummond, [doi](https://link.springer.com/chapter/10.1007/11744023_34).

The found features from this crate are equivalent to the output of OpenCV. Runtime is almost half of OpenCV with identical image and parameters.

It makes heavy use of the AVX2 instruction set to achieve the highest possible throughput.

## Algorithm:
  Extremely concise version of the algorithm.
  - Compare points on a circle against the center point, circle has 16 points.
  - Check if points on the circle exceed center point by a threshold.
  Definition of the paper is, let a cirle point be p and center of the circle c.
    - darker: `p <= c - t`
    - similar: `c - t < p < c + t`
    - brigher: `c + t <= p`
  - If there are more than n consecutive pixels on the circle that are darker, the center point is a feature.
  - If there are more than n consecutive pixels on the circle that are lighter, the center point is a feature.
  - If non maximum suppression is enabled, calculate that for each candidate feature, they only become features if they are the strongest feature compared to their 8 neighbours.
  

## Overall approach is:
  - Iterate through each row.
    - Iterate through the row in blocks of 16 pixels. The center pixels.
    - For each center, determine whether the cardinal directions exceed the threshold.
      For n >= 12: 3/4 need to match.
      For n > 9 && n < 12: 2/4 need to match.
      If these match, set a bit to mark it as a potential.
    - Iterate throught potentials and perform the thorough check.
      - Thorough check uses two gather operations to retrieve the entire circle's pixels
      - Wrangle these points (conveniently 16 of them) into a single 128 bit vector.
      - Determine if points exceed the threshold above or below the center point.
      - Reduce the data to a single integer, each bit representing whether the bound is exceeded.
      - Use popcnt to determine if it is a positive or negative keypoint
      - Iterate throught the correct integer, checking if the correct number of consecutive
        value exceeds has been found.

## Tests
Tests ran on my system (i7-4770TE, from 2014) against a 1920 x 1080 grayscale image from a
computer game.

##### Results without non-maximum supression:
  - OpenCV takes 18'ish milliseconds to run with a threshold of 16, 9/16 consecutive, no nonmax supression. This finds 23184 keypoints.
  - This implementation takes 10'ish milliseconds, with the same parameters. And finds the same 23184 keypoints.
##### Results with non-maximum supression:
  - OpenCV takes 31'ish milliseconds to run with a threshold of 16, 9/16 consecutive, nonmax supression using maximum 't' for which it is a keypoint. This finds 7646 keypoints.
  - This implementation takes 14'ish milliseconds, with the same parameters. And finds the same 7646 keypoints.

## Remarks
  - The current non maximum supression score function is the maximum 't' for which a feature would still be a feature.
    According to the paper this score function is often very similar between pixels in an image.