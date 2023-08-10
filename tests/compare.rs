use feature_detector_fast::{
    fast_simd, opencv_compat, util, FastConfig, FastPoint, NonMaximalSuppression,
};

fn hash_result(points: &[FastPoint]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut s = DefaultHasher::new();
    points.hash(&mut s);
    s.finish()
}
fn hash_slice_u8(d: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut s = DefaultHasher::new();
    d.hash(&mut s);
    s.finish()
}

#[test]
pub fn run_test() -> Result<(), Box<dyn std::error::Error>> {
    let input_image_file =
        std::env::var("INPUT_FILE").unwrap_or("media/Screenshot315_torch_grey.png".to_string());

    let image_path = std::path::PathBuf::from(&input_image_file);
    let orig_image = image::open(&image_path)
        .expect(&format!("could not load image at {:?}", input_image_file))
        .to_rgb8();

    // let luma_view = util::Rgb8ToLuma16View::new(&orig_image);
    let luma_view = image::DynamicImage::ImageRgb8(orig_image.clone()).to_luma8();
    let _ = luma_view.save("/tmp/rust_grey.png")?;

    let circle_image = opencv_compat::make_circle_image();
    let _ = circle_image.save("/tmp/circle_image.png")?;

    fn compare_simd_normal(
        luma_view: &image::GrayImage,
        config: &FastConfig,
        name: &str,
    ) -> Result<Vec<FastPoint>, Box<dyn std::error::Error>> {
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

    // This is only here since I always test on the same image, these hashes bail out in case anything
    // changes in the results. ../screenshots/Screenshot315_grey.png
    let hash_keypoints = hash_result(&keypoints);
    println!("Hash of keypoints: 0x{hash_keypoints:x}");
    if hash_slice_u8(orig_image.as_raw()) == 0x8444a9356505ecab
        && hash_keypoints != 0x8bf9cd0f9ca9ebec
    {
        panic!("Not hash equal against default test image.");
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
