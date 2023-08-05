pub mod fast;
pub mod fast_simd;
pub mod util;

use image;

pub fn run_test() -> Result<(), Box<dyn std::error::Error>> {
    let input_image_file = std::env::args().nth(1).expect("no image file specified");

    let image_path = std::path::PathBuf::from(&input_image_file);
    let orig_image = image::open(&image_path)
        .expect(&format!("could not load image at {:?}", input_image_file))
        .to_rgb8();

    // let luma_view = util::Rgb8ToLuma16View::new(&orig_image);
    let luma_view = image::DynamicImage::ImageRgb8(orig_image.clone()).to_luma8();
    let _ = luma_view.save("/tmp/rust_grey.png")?;

    let circle_image = fast::fast_detector16::make_circle_image();
    let _ = circle_image.save("/tmp/circle_image.png")?;

    let config = fast::FastConfig {
        threshold: 16,
        count: 9,
        non_maximal_supression: false,
    };

    let start = std::time::Instant::now();
    let mut keypoints_simd = fast_simd::detector(&luma_view, &config);
    println!("simd is: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let mut keypoints = fast::detector(&luma_view, &config);
    println!("normal is: {:?}", start.elapsed());

    if keypoints_simd != keypoints {
        println!("Keypoints not identical");
    }
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

    Ok(())
}
