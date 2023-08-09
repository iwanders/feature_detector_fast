/*!
    This implements the FAST feature detector from Rosten & Drummond, 2006.

    It follows the description of rosten2006.pdf "LNCS 3951 - Machine Learning for High-Speed Corner Detection".

    The simd implementation is faster than OpenCV's implementation, by almost a factor of two. See the comments in [fast_simd].
*/

pub mod opencv_compat;
pub mod util;

#[cfg(any(doc, all(any(target_arch = "x86_64"), target_feature = "avx2")))]
pub mod fast_simd;

use image;

#[derive(Copy, Debug, Clone, Eq, PartialEq, Hash, Default)]
/// A feature point at an image position.
pub struct FastPoint {
    pub x: u32,
    pub y: u32,
}

/// Modes of non maximal suppression, this filters feature candidates to ensure only the strongest
/// points from the 8 neighbours are selected. Example numbers are given for a 1080p image, count=9
/// threshold=16, using this crate and the simd implementation.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum NonMaximalSuppression {
    /// No non maximal suppression, all features satisfying the consecutive circle threshold are
    /// returned as keypoints. Example runtime: 10ms.
    Off,
    /// Use the maximum t for which this feature would still be a feature. This is what OpenCV uses.
    /// Example runtime 15ms
    MaxThreshold,
    /// Take the absolute sum of all pixels in the dark or light set. This is what the authors
    /// recommend, and it is extremely cheap to calculate. Example runtime 13ms.
    SumAbsolute,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FastConfig {
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

fn hash_result(points: &[FastPoint]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut s = DefaultHasher::new();
    points.hash(&mut s);
    s.finish()
}

pub fn run_test() -> Result<(), Box<dyn std::error::Error>> {
    let input_image_file = std::env::args().nth(1).expect("no image file specified");

    let image_path = std::path::PathBuf::from(&input_image_file);
    let orig_image = image::open(&image_path)
        .expect(&format!("could not load image at {:?}", input_image_file))
        .to_rgb8();

    // let luma_view = util::Rgb8ToLuma16View::new(&orig_image);
    let luma_view = image::DynamicImage::ImageRgb8(orig_image.clone()).to_luma8();
    let _ = luma_view.save("/tmp/rust_grey.png")?;

    let circle_image = opencv_compat::make_circle_image();
    let _ = circle_image.save("/tmp/circle_image.png")?;

    fn compare_simd_normal(luma_view: &image::GrayImage, config: &FastConfig, name: &str) -> Result<Vec<FastPoint>, Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();
        let keypoints_simd = fast_simd::detector(&luma_view, &config);
        println!("{name} simd  : {:?}", start.elapsed());

        let start = std::time::Instant::now();
        let keypoints = opencv_compat::detector(&luma_view, &config);
        println!("{name} normal: {:?}", start.elapsed());
        {
            let mut rgb_owned = image::DynamicImage::ImageLuma8(luma_view.clone()).to_rgb8();
            for kp in keypoints.iter() {
                util::draw_plus_sized(&mut rgb_owned, (kp.x, kp.y), util::RED, 1);
            }
            let _ = rgb_owned.save(format!("/tmp/with_rust_{name}.png"));
        }

        if keypoints_simd != keypoints {
            panic!("Keypoints not identical");
        }
        println!("{name} Found {} keypoints", keypoints.len());
        Ok(keypoints_simd)
    }


    let config = FastConfig {
        threshold: 16,
        count: 9,
        non_maximal_supression: NonMaximalSuppression::Off,
    };
    compare_simd_normal(&luma_view, &config, "non_max_suppression_t16_c_9")?;


    println!("--");
    let config = FastConfig {
        threshold: 16,
        count: 9,
        non_maximal_supression: NonMaximalSuppression::MaxThreshold,
    };
    let keypoints = compare_simd_normal(&luma_view, &config, "max_threshold_t16_c_9")?;

    let hash_keypoints = hash_result(&keypoints);
    println!("Hash of keypoints: 0x{hash_keypoints:x}");
    if hash_keypoints != 0x8bf9cd0f9ca9ebec {
        panic!("Not hash equal");
    }

    println!("--");

    let config = FastConfig {
        threshold: 16,
        count: 9,
        non_maximal_supression: NonMaximalSuppression::SumAbsolute,
    };
    compare_simd_normal(&luma_view, &config, "sum_absolute_t16_c_9")?;


    println!("--");
    let config = FastConfig {
        threshold: 16,
        count: 12,
        non_maximal_supression: NonMaximalSuppression::SumAbsolute,
    };
    compare_simd_normal(&luma_view, &config, "sum_absolute_t16_c_12")?;

    println!("--");
    let config = FastConfig {
        threshold: 32,
        count: 12,
        non_maximal_supression: NonMaximalSuppression::SumAbsolute,
    };
    compare_simd_normal(&luma_view, &config, "sum_absolute_t32_c_16")?;



    Ok(())
}
