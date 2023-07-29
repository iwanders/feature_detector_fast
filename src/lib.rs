pub mod fast;
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

    let config = fast::FastConfig {
        threshold: 64,
        count: 9,
    };

    let mut r = vec![];

    // if let Some(p) = fast::fast_detector16::detect(
        // (723, 258),
        // &luma_view,
        // config.thresshold as u16 * 3,
        // config.count,
    // ) {
        // r.push(p);
    // }

    // if let Some(p) = fast::fast_detector16::detect(
        // (487, 254),
        // &luma_view,
        // config.thresshold as u16 * 3,
        // config.count,
    // ) {
        // r.push(p);
    // }
    let start = std::time::Instant::now();
    let mut keypoints = fast::detector(&luma_view, &config);
    // let mut keypoints = fast::detector12(&luma_view, &config);

    let circle_image = fast::fast_detector16::make_circle_image();
    let _ = circle_image.save("/tmp/circle_image.png")?;
    

    let duration = start.elapsed();
    println!("Time elapsed in expensive_function() is: {:?}", duration);
    r.extend(&mut keypoints.drain(..));

    // let keypoints = if kp.is_some() {vec![kp.unwrap()]} else {vec![]};

    // let owned = image::ImageBuffer::<image::Luma<u16>, Vec<_>>::from(&luma_view);
    // let grey_owned = luma_view.to_grey();

    let mut rgb_owned = image::DynamicImage::ImageLuma8(luma_view.clone()).to_rgb8();
    // let mut rgb_owned = grey_owned.to_rgb8();

    for kp in r.iter() {
        // println!("kp: {kp:?}");
        util::draw_plus_sized(&mut rgb_owned, (kp.x, kp.y), util::RED, 1);
    }

    let _ = rgb_owned.save("/tmp/grey.png");

    Ok(())
}
