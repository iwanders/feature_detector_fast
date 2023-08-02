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

/*

     15 0 1
   14       2
 13           3
 12    +      4
 11           5
   10       6
     9  8  7
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

    pub fn detect(image: &image::GrayImage, t: u8, consecutive: u8) -> Vec<FastPoint> {
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

        unsafe {
            let indices = _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(
                &circle_offset[0],
            ));
            for y in 3..(height - 3) {
                // we should probably do something smarter than this.
                //     15 0 1
                //   14       2
                // 13           3
                // 12    +      4
                // 11           5
                //   10       6
                //     9  8  7
                // What about checking the entire row of 12-4 and and only for those where
                // 12 and 4 indicate possible corner check anything off the row?
                    
                for x in 3..(width - 3) {
                    let base_offset = (y * width + x) as i32;

                    let base_v = data[base_offset as usize];

                    // pub unsafe fn _mm_loadu_si64(mem_addr: *const u8) -> __m128i

                    // Perform a single gather to obtain the first 8 indices.
                    // core::arch::x86_64::_mm256_i32gather_epi32
                    // Gather 32-bit integers from memory using 32-bit indices. 32-bit elements are
                    // loaded from addresses starting at base_addr and offset by each 32-bit element in
                    // vindex (each index is scaled by the factor in scale).
                    // Gathered elements are merged into dst. scale should be 1, 2, 4 or 8.

                    // We have pixels that are offset by one, so our scale is one.
                    //
                    // std::arch::x86_64::_mm256_i32gather_epi32(lookup_base, unpacked_in_32, SCALE);
                    /*
                        result[31:0] = mem[base+vindex[31:0]*scale];
                        result[63:32] = mem[base+vindex[63:32]*scale];
                        result[95:64] = mem[base+vindex[95:64]*scale];
                        result127:96] = mem[base+vindex[127:96]*scale];

                        result[159:128] = mem[base+vindex[159:128]*scale];
                        result[191:160] = mem[base+vindex[191:160]*scale];
                        result[223:192] = mem[base+vindex[223:192]*scale];
                        result[255:224] = mem[base+vindex[255:224]*scale];
                    */
                    // vindex is already inside circle_offset, so we can perform one fell swoop to land
                    // us the first 8 indices.
                    const SCALE: i32 = 1;
                    let lookup_base =
                        std::mem::transmute::<_, *const i32>(&data[base_offset as usize]);
                    let obtained = _mm256_i32gather_epi32(lookup_base, indices, SCALE);

                    /*
                    // after the gather, we end up with
                    // v0 0 0 0 v1 0 0 0 v2 0 0 0 v3 0 0 0 | v4 0 0 0 v5 0 0 0 v6 0 0 0 v7
                    let mask = _mm256_set_epi64x(
                        0, // discarded anyway
                        // on zero'th byte, we want the 0 index, second byte, index 4, third; 8th...
                        0x0c080400, 0, // discarded anyway
                        0x0c080400,
                    );
                    // we do this magic shuffle back to collapse that into
                    let left_u32_per_lane = _mm256_shuffle_epi8(obtained, mask);
                    // v0 v1 v2 v3 0000000 | v4 v5 v6 v7
                    let lower = _mm256_extract_epi32(left_u32_per_lane, 0);
                    let higher = _mm256_extract_epi32(left_u32_per_lane, 4);

                    // Finally, we can store that into a single array for indexing later.
                    let mut retrievable = [0u8; 8];
                    retrievable[0..4].copy_from_slice(&lower.to_le_bytes());
                    retrievable[4..].copy_from_slice(&higher.to_le_bytes());
                    */
                    

                    let delta_f = |index: usize| {
                        let pixel_v = data[(base_offset + circle_offset[index]) as usize];
                        let delta = base_v as i16 - pixel_v as i16;
                        delta
                    };
                    let p_f = |index: usize| data[(base_offset + circle_offset[index]) as usize];

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
        }
        r
    }
}
pub fn detector(img: &image::GrayImage, config: &crate::fast::FastConfig) -> Vec<FastPoint> {
    fast_detector16::detect(img, config.threshold, config.count)
}
