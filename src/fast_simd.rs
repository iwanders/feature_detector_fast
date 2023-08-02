use crate::fast::FastPoint;
use image::{GenericImageView, Luma};

/*
    avx
    avx2
    sse
    sse2
    sse3
    sse4_1
    sse4_2

    // https://doc.rust-lang.org/stable/core/arch/x86_64/struct.__m128i.html
    https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=MMX&avxnewtechs=AVX,AVX2&ssetechs=SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2

*/

#[cfg(all(any(target_arch = "x86_64"), target_feature = "avx2"))]
pub mod fast_detector16 {

    use std::arch::x86_64::*;
    unsafe fn pi(input: &__m128i) -> String {
        let v: [u8; 16] = [0; 16];
        _mm_storeu_si128(v.as_ptr() as *mut _, *input);
        format!("{:02X?}", v)
    }
    // Print long simd type
    unsafe fn pl(input: &__m256i) -> String {
        let v: [u8; 32] = [0; 32];
        _mm256_storeu_si256(v.as_ptr() as *mut _, *input);
        format!("{:02X?}", v)
    }

    const DO_PRINTS: bool = false;

    #[allow(unused_macros)]
    macro_rules! trace {
        () => (if DO_PRINTS {println!("\n");});
        ($($arg:tt)*) => {
            if DO_PRINTS {
                println!($($arg)*);
            }
        }
    }


    use super::*;

    pub const NORTH: usize = 0;
    pub const EAST: usize = 4;
    pub const SOUTH: usize = 8;
    pub const WEST: usize = 12;

    /// The circle with 16 pixels.
    pub const fn circle() -> [(i32, i32); 16] {
        [
            (0, -3),
            (1, -3),
            (2, -2),
            (3, -1),
            (3, -0),
            (3, 1),
            (2, 2),
            (1, 3),
            (0, 3),
            (-1, 3),
            (-2, 2),
            (-3, 1),
            (-3, -0),
            (-3, -1),
            (-2, -2),
            (-1, -3),
        ]
    }

    pub const fn point(index: u8) -> (i32, i32) {
        circle()[index as usize % circle().len()]
    }

    pub fn make_circle_image() -> image::RgbImage {
        const BLUE: image::Rgb<u8> = image::Rgb([0u8, 0u8, 255u8]);
        let mut image = image::RgbImage::new(32, 32);
        for (dx, dy) in circle().iter() {
            image.put_pixel((16i32 + dx) as u32, (16i32 + dy) as u32, BLUE)
        }
        image
    }

    pub fn detect(
        image: &image::GrayImage,
        t: u8,
        consecutive: u8,
    ) -> Vec<FastPoint> {
        let height = image.height();
        let width = image.width();
        let t = t as i16;
        // exists range n where all entries different than p - t.
        let mut r = vec![];

        let data = image.as_raw();

        // calculate the circle offsets for the data once.
        let mut circle_offset = [0i32; 16];
        for (i, (x, y)) in circle().iter().enumerate() {
            circle_offset[i] = *y * width as i32 + *x;
        }
        

        for y in 3..(height - 3) {
            for x in 3..(width - 3) {
                let base_offset = (y * width + x) as i32;

                let base_v = data[base_offset as usize];

                let delta_f = |index: usize| {
                    let pixel_v = data[(base_offset + circle_offset[index]) as usize];
                    let delta = base_v as i16 - pixel_v as i16;
                    delta
                };
                let p_f = |index: usize| {
                    data[(base_offset + circle_offset[index]) as usize]
                };

                const COUNT: usize = circle().len() as usize;
                let mut neg = [false; COUNT];
                let mut pos = [false; COUNT];
                for i in 0..COUNT {
                    let d = delta_f(i);
                    let a = d.abs();
                    neg[i] = d < 0 && a > t as i16;
                    pos[i] = d > 0 && a > t as i16;
                }

                if DO_PRINTS && false {
                    for i in 0..COUNT {
                        print!("  {} ", delta_f(i));
                    }
                    println!("  t: {t}");
                    for i in 0..COUNT {
                        print!("  {} ", p_f(i));
                    }
                    println!("  pixels");
                    println!("  neg: {neg:?}");
                    println!("  pos: {pos:?}");
                }

                // There's probably a way more efficient way of doing this rotation.
                for s in 0..COUNT {
                    let n = neg
                        .iter()
                        .cycle()
                        .skip(s)
                        .take(COUNT)
                        .take_while(|t| **t)
                        .count()
                        >= consecutive as usize;
                    let p = pos
                        .iter()
                        .cycle()
                        .skip(s)
                        .take(COUNT)
                        .take_while(|t| **t)
                        .count()
                        >= consecutive as usize;

                    if n || p {
                        if DO_PRINTS {
                            println!("  Succceed by p: {p}, n: {n} at s {s}");
                        }
                        r.push(FastPoint { x, y });
                    }
                }
            }
        }

        r
    }

}
pub fn detector(
    img: &image::GrayImage,
    config: &crate::fast::FastConfig,
) -> Vec<FastPoint> {

    fast_detector16::detect(img, config.threshold, config.count)
}
