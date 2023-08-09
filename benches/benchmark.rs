use criterion::{black_box, criterion_group, criterion_main, Criterion};
use feature_detector_fast::{fast_simd, FastConfig, NonMaximalSuppression};
use image;

pub fn criterion_benchmark(c: &mut Criterion) {
    let input_image_file =
        std::env::var("INPUT_FILE").unwrap_or("media/Screenshot315_torch_grey.png".to_string());
    // let input_image_file = std::env::args().nth(1).expect("no image file specified");

    let image_path = std::path::PathBuf::from(&input_image_file);
    let orig_image = image::open(&image_path)
        .expect(&format!("could not load image at {:?}", input_image_file))
        .to_rgb8();

    let config = FastConfig {
        threshold: 16,
        count: 9,
        non_maximal_supression: NonMaximalSuppression::Off,
    };
    // let luma_view = util::Rgb8ToLuma16View::new(&orig_image);
    let luma_view = image::DynamicImage::ImageRgb8(orig_image.clone()).to_luma8();

    c.bench_function("simd", |b| {
        b.iter(|| {
            let keypoints_simd = fast_simd::detector(&luma_view, &config);
            black_box(keypoints_simd);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
