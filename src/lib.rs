pub mod opencv_compat;

#[cfg(any(doc, all(any(target_arch = "x86_64"), target_feature = "avx2")))]
pub mod fast_simd;

pub mod util;

use image;


#[derive(Copy, Debug, Clone, Eq, PartialEq, Hash)]
pub struct FastPoint {
    pub x: u32,
    pub y: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct FastConfig {
    /// Value to be exceeded.
    pub threshold: u8,

    /// Count of consecutive pixels
    pub count: u8,

    /// Whether to use non maximal suprresion.
    pub non_maximal_supression: bool,
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

    let config = FastConfig {
        threshold: 16,
        count: 9,
        non_maximal_supression: true,
    };

    let start = std::time::Instant::now();
    let keypoints_simd = fast_simd::detector(&luma_view, &config);
    println!("simd is: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let keypoints = opencv_compat::detector(&luma_view, &config);
    println!("normal is: {:?}", start.elapsed());

    if keypoints_simd != keypoints {
        println!("Keypoints not identical");
    }

    let hash_keypoints = hash_result(&keypoints);

    println!("Found {} keypoints", keypoints.len());
    // keypoints = keypoints_simd;

    // let keypoints = if kp.is_some() {vec![kp.unwrap()]} else {vec![]};

    // let owned = image::ImageBuffer::<image::Luma<u16>, Vec<_>>::from(&luma_view);
    // let grey_owned = luma_view.to_grey();

    let mut rgb_owned = image::DynamicImage::ImageLuma8(luma_view.clone()).to_rgb8();
    for kp in keypoints_simd.iter() {
        util::draw_plus_sized(&mut rgb_owned, (kp.x, kp.y), util::RED, 1);
    }
    let _ = rgb_owned.save("/tmp/with_rust_simd.png");

    let mut rgb_owned = image::DynamicImage::ImageLuma8(luma_view.clone()).to_rgb8();
    for kp in keypoints.iter() {
        util::draw_plus_sized(&mut rgb_owned, (kp.x, kp.y), util::RED, 1);
    }
    let _ = rgb_owned.save("/tmp/with_rust.png");

    println!("Hash of keypoints: 0x{hash_keypoints:x}");
    if hash_keypoints != 0x8bf9cd0f9ca9ebec {
        panic!("Not hash equal");
    }

    Ok(())
}
