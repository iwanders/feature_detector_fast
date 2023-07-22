
mod util;

use image;

pub fn run_test() -> Result<(), Box<dyn std::error::Error>> {
    let input_image_file = std::env::args().nth(1).expect("no image file specified");

    let image_path = std::path::PathBuf::from(&input_image_file);
    let orig_image = image::open(&image_path)
        .expect(&format!("could not load image at {:?}", input_image_file))
        .to_rgb8();

    let luma_view = util::Rgb8ToLuma16View::new(&orig_image);

    // let owned = image::ImageBuffer::<image::Luma<u16>, Vec<_>>::from(&luma_view);
    let grey_owned = luma_view.to_grey();

    let _ = grey_owned.save("/tmp/grey.png");

    Ok(())
}