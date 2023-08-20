/*!
    This implements the FAST feature detector from Rosten & Drummond, 2006.

    It follows the description of rosten2006.pdf "LNCS 3951 - Machine Learning for High-Speed Corner Detection".

    The simd implementation is faster than OpenCV's implementation, by a factor of ~three. See the comments in [fast_simd].
*/

pub mod opencv_compat;
pub mod util;

#[cfg(any(doc, all(any(target_arch = "x86_64"), target_feature = "avx2")))]
pub mod fast_simd;

#[derive(Copy, Debug, Clone, Eq, PartialEq, Hash, Default)]
/// A feature point at an image position.
pub struct Point {
    pub x: u32,
    pub y: u32,
}

/// Modes of non maximal suppression, this filters feature candidates to ensure only the strongest
/// points from the 8 neighbours are selected. Example numbers are given for a 1080p image, count=9
/// threshold=16, using this crate and the simd implementation.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum NonMaximalSuppression {
    /// No non maximal suppression, all features satisfying the consecutive circle threshold are
    /// returned as keypoints. Example runtime: 6ms.
    Off,
    /// Use the maximum t for which this feature would still be a feature. This is what OpenCV uses.
    /// Example runtime 11ms
    MaxThreshold,
    /// Take the absolute sum of all pixels in the dark or light set. This is what the authors
    /// recommend, and it is extremely cheap to calculate. Example runtime 9ms.
    SumAbsolute,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// Configuration struct for the FAST feature detector.
pub struct Config {
    /// The threshold by which the values on the circle must exceed the center point to be counted
    /// towards the consecutive count.
    pub threshold: u8,

    /// The minimum count of consecutive pixels in the circle.
    /// Allowed values are count >= 9 && count <= 16, for count >= 12, a third cardinal direction
    /// must be true, so any count above 12 enables further speedups.
    pub count: u8,

    /// Whether to use non maximal suppression.
    pub non_maximal_supression: NonMaximalSuppression,
}

impl Config {
    /// Method access to run the detector.
    pub fn detect(&self, img: &image::GrayImage) -> Vec<Point> {
        fast_simd::detector(img, self)
    }
}

/// Function to perform the FAST keypoint detection.
pub fn detect(img: &image::GrayImage, config: &Config) -> Vec<Point> {
    fast_simd::detector(img, config)
}
