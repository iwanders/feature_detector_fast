use feature_detector_fast::{fast_simd, util, FastConfig, NonMaximalSuppression};
use image;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().len() == 1
        || (std::env::args().len() == 2 && std::env::args().nth(1) == Some("--help".to_string()))
    {
        println!("cargo r --release -- <input> [output(default; /tmp/output.png)] [threshold(default: 16)] [count(default:9)] [non_maximal_suppression:off|sum_absolute|max_threshold (default: max_threshold)]");
        println!(" arguments required left to right.");
        return Ok(());
    }

    let input_image_file = std::env::args().nth(1).expect("no image file specified");
    let output_image_file = std::env::args()
        .nth(2)
        .unwrap_or("/tmp/output.png".to_string());
    let threshold = std::env::args()
        .nth(3)
        .unwrap_or("16".to_string())
        .parse::<u8>()
        .expect("failed to parse threshold");
    let count = std::env::args()
        .nth(4)
        .unwrap_or("9".to_string())
        .parse::<u8>()
        .expect("failed to parse count");
    let non_maximal_supression = match std::env::args()
        .nth(5)
        .unwrap_or("sum_absolute".to_string())
        .as_str()
    {
        "off" => NonMaximalSuppression::Off,
        "sum_absolute" => NonMaximalSuppression::SumAbsolute,
        "max_threshold" => NonMaximalSuppression::MaxThreshold,
        _ => panic!("unknown non maximal, support: off, sum_absolute, max_threshold"),
    };

    let image_path = std::path::PathBuf::from(&input_image_file);
    let orig_image = image::open(&image_path)
        .expect(&format!("could not load image at {:?}", input_image_file))
        .to_rgb8();

    // let luma_view = util::Rgb8ToLuma16View::new(&orig_image);
    let luma_view = image::DynamicImage::ImageRgb8(orig_image.clone()).to_luma8();
    // let _ = luma_view.save("/tmp/rust_grey.png")?;

    let config = FastConfig {
        threshold,
        count,
        non_maximal_supression,
    };
    let start = std::time::Instant::now();
    let keypoints = fast_simd::detector(&luma_view, &config);
    println!(
        "Took: {:?}, found {} keypoints",
        start.elapsed(),
        keypoints.len()
    );

    let mut rgb_owned = image::DynamicImage::ImageLuma8(luma_view.clone()).to_rgb8();
    for kp in keypoints.iter() {
        util::draw_plus_sized(&mut rgb_owned, (kp.x, kp.y), util::RED, 1);
    }
    let _ = rgb_owned.save(&output_image_file)?;

    Ok(())
}
