
mod util;
mod fast;

use image;

pub fn run_test() -> Result<(), Box<dyn std::error::Error>> {
    let input_image_file = std::env::args().nth(1).expect("no image file specified");

    let image_path = std::path::PathBuf::from(&input_image_file);
    let orig_image = image::open(&image_path)
        .expect(&format!("could not load image at {:?}", input_image_file))
        .to_rgb8();

    let luma_view = util::Rgb8ToLuma16View::new(&orig_image);

    let config = fast::FastConfig {
        thresshold: 32,
        count: 12,
    };

    let keypoints = fast::detector(&orig_image, &config);

    // let owned = image::ImageBuffer::<image::Luma<u16>, Vec<_>>::from(&luma_view);
    let grey_owned = luma_view.to_grey();

    let mut rgb_owned = image::DynamicImage::ImageLuma8(grey_owned).to_rgb8();
    // let mut rgb_owned = grey_owned.to_rgb8();

    for kp in keypoints.iter() {
        util::draw_plus(&mut rgb_owned, (kp.x, kp.y), util::RED);
    }

    let _ = rgb_owned.save("/tmp/grey.png");

    Ok(())
}