# Feature Detector FAST

This is a highly optimised implementation of the [FAST feature detector](https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test). As described in:

> Rosten, E., Drummond, T. (2006). Machine Learning for High-Speed Corner Detection. In: Leonardis, A., Bischof, H., Pinz, A. (eds) Computer Vision – ECCV 2006. ECCV 2006. Lecture Notes in Computer Science, vol 3951. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11744023_34

The features detected by this implementation are identical to those found by OpenCV for the same image and parameters. This implementation makes heavy use of the AVX2 instruction set to achieve the highest possible throughput; runtime is almost ~~half~~ a third of OpenCV, while providing the exact same results.

In addition to this, the minimum consecutive pixel count is configurable (as opposed to OpenCV), with 9 being the lower bound, up to the entire circle of 16 pixels. At 12 an additional cardinal direction check is possible, allowing for even more speedups in images where a consecutive count of `n >= 12` is feasible.

## Example
Example images to give an idea (left to right): RGB input, Grayscale, features detected with `threshold = 16`, `consecutive >= 9`, and the same with non-max suppression.

<p align="middle">
  <img src="/media/Screenshot315_torch.png" width="20%" />
  <img src="/media/Screenshot315_torch_grey.png" width="20%" /> 
  <img src="/media/with_rust_threshold_16_consecutive_9.png" width="20%" />
  <img src="/media/with_rust_threshold_16_consecutive_9.png_nonmax.png" width="20%" />
</p>

The `consecutive >= 9` was chosen for this example because that is the only flavour OpenCV implements, its results are [here](/media/with_opencv_threshold_16_type_9_16.png) and [here](with_opencv_threshold_16_type_9_16_nonmax.png).

## Algorithm:
  Extremely concise version of the algorithm.
  - Points on a circle are compared against the center point, the circle has 16 points.
  - Check if points on the circle exceed center point by a threshold.
  Definition from the paper is, let a cirle point be p and center of the circle c.
    - darker: `p <= c - t`
    - similar: `c - t < p < c + t`
    - brigher: `c + t <= p`
  - If there are more than n consecutive pixels on the circle that are darker, the center point is a feature.
  - If there are more than n consecutive pixels on the circle that are lighter, the center point is a feature.
  - If non maximum suppression is enabled, calculate that for each candidate feature, candidates only become features if they are the strongest feature compared to their 8 neighbours.
  

## Implementation

High level overview of the implementation. Images are grayscale, one byte per pixel and row major.

  - Iterate through each row.
    - Iterate through the row in blocks of 16 pixels. The center pixels.
    - For each block, also load 16 pixels from each cardinal direction.
    - For each center, determine whether the cardinal directions exceed the threshold.
      For n >= 12: 3 out of 4 need to match.
      For n >= 9 && n < 12: 2 out of 4 need to match.
      If these match, set a bit to mark it as a potential.
    - Iterate through the potentials and perform the thorough check.
      - Thorough check uses two gather operations to retrieve the entire circle's pixels
      - Wrangle these points (conveniently 16 of them) into a single 128 bit vector.
      - Determine if points exceed the threshold above or below the center point, creates the above and below vectors.
        - Use a mask that has n consecutive bytes set high. This mask is rotated by one byte 16 times, at each iteration, check if the masked above or below vector is completely populated with zeros. If it is, the required number of consecutive points exceeding the threshold has been found.
      - If non max suppression is enabled, the features' score values are stored in an intermediate vector, which is finalised after the next row is calculated.

## Performance
Tests ran on my system (i7-4770TE, from 2014) against a 1920 x 1080 grayscale image from a computer game. Numbers are obtained using [criterion 0.5.1](https://docs.rs/criterion/0.5.1/criterion/index.html), notation is [[left bound, mean, right bound]](https://bheisler.github.io/criterion.rs/book/user_guide/command_line_output.html#time) of a 95% confidence interval.

##### Results without non-maximum supression:
  - OpenCV takes 18'ish milliseconds to run with a threshold of 16, 9/16 consecutive, no nonmax supression. This finds 23184 keypoints.
  - This implementation runs in `[5.2950 ms 5.3381 ms 5.3857 ms]`, with the same parameters, and finds the same 23184 keypoints.
##### Results with max threshold non-maximum supression:
  - OpenCV takes 31'ish milliseconds to run with a threshold of 16, 9/16 consecutive, nonmax supression using maximum 't' for which it is a keypoint. This finds 7646 keypoints.
  - This implementation runs in `[8.6836 ms 8.7080 ms 8.7357 ms]` milliseconds, with the same parameters, and finds the same 7646 keypoints.
##### Results with sum of absolute difference non-maximum supression:
  - OpenCV does not implement this score function.
  - This implementation runs in `[7.2208 ms 7.2343 ms 7.2494 ms]`. It finds 8307 keypoints.

## Remarks
  - The rust [image 0.24.6](https://docs.rs/image/0.24.6/image/index.html) crate does grey scale conversion differently than OpenCV does. Be sure to make it grayscale with an editor for comparisons.


## License
License is `BSD-3-Clause`.
