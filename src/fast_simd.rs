//!
//! This implements the FAST feature detector from Rosten & Drummond, 2006.
//!
//! It makes heavy use of the AVX2 instruction set.
//! 
//!   - The circle has 16 points, which fits in a single 128 bit instruction lane.
//! 
//! Definition of the paper is, let a cirle point be p and center of the circle c.
//!     darker: p <= c - t
//!     similar: c - t < p < c + t
//!     brigher: c + t <= p
//!
//! Overall approach is:
//!   - Iterate through each row.
//!     - Iterate through the row in blocks of 16 pixels. The center pixels.
//!     - For each center, determine whether the cardinal directions exceed the threshold.
//!       For n >= 12: 3/4 need to match.
//!       For n > 9 && n < 12: 2/4 need to match.
//!       If these match, set a bit to mark it as a potential.
//!     - Iterate throught potentials and perform the thorough check.
//!       - Thorough check uses two gather operations to retrieve the entire circle's pixels
//!       - Wrangle these points into a single 128 bit vector.
//!       - Determine if points exceed the threshold above or below the center point.
//!       - Reduce the data to a single integer, each bit representing whether the bound is exceeded.
//!       - Use popcnt to determine if it is a positive or negative keypoint
//!       - Iterate throught the correct integer, checking if the correct number of consecutive
//!         value exceeds has been found.
//!
//! Tests ran on my system (i7-4770TE, from 2014) against a 1920 x 1080 grayscale image from a
//! computer game.
//! 
//! OpenCV takes 18'ish milliseconds to run with a threshold of 16, 9/16 consecutive, no nonmax supression. This finds 23184 keypoints.
//! This implementation takes 10'ish milliseconds, with the same parameters. And finds the same 23184 keypoints.
//!
//!


use crate::{FastPoint, FastConfig};
use std::arch::x86_64::*;

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

#[allow(dead_code)]
/// Print a vector of m128 type.
unsafe fn pi(input: &__m128i) -> String {
    let v: [u8; 16] = [0; 16];
    _mm_storeu_si128(v.as_ptr() as *mut _, *input);
    format!("{:02X?}", v)
}

#[allow(dead_code)]
/// Print a vector of m256 type.
unsafe fn pl(input: &__m256i) -> String {
    let v: [u8; 32] = [0; 32];
    _mm256_storeu_si256(v.as_ptr() as *mut _, *input);
    format!("{:02X?}", v)
}

const DO_PRINTS: bool = false;

#[allow(unused_macros)]
/// Helper print macro that can be enabled or disabled.
macro_rules! trace {
    () => (if DO_PRINTS {println!("\n");});
    ($($arg:tt)*) => {
        if DO_PRINTS {
            println!($($arg)*);
        }
    }
}

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

pub type CircleOffsets = [i32; 16];
/// Calculate the offets in the memory block for an image of a certain width.
pub fn calculate_offsets(width: u32) -> CircleOffsets {
    let mut circle_offset = [0i32; 16];
    for (i, (x, y)) in circle().iter().enumerate() {
        circle_offset[i] = *y * width as i32 + *x;
    }
    circle_offset
}

/// Compare u8 types in a m128 vector.
///  https://stackoverflow.com/a/24234695
unsafe fn _mm_cmpgt_epu8(a: __m128i, b: __m128i) -> __m128i {
    // ah, this is a signed comparison...
    // https://stackoverflow.com/a/24234695
    _mm_cmpgt_epi8(
        _mm_xor_si128(a, _mm_set1_epi8(-128)), // range-shift to unsigned
        _mm_xor_si128(b, _mm_set1_epi8(-128)),
    )
}

// This inline keyword actually makes a difference.
#[inline]
/// Determine whether there's a keypoint at the provided location.
unsafe fn determine_keypoint(
    data: &[u8],
    circle_offset: &CircleOffsets,
    width: u32,
    p: (u32, u32),
    t: u8,
    consecutive: u8,
) -> Option<FastPoint> {
    trace!("\n\nDetermine keypoint at {p:?}");
    // Some shorthands
    let xx = p.0;
    let y = p.1;
    let base_offset = (y * width + xx) as i32;
    let base_v = data[base_offset as usize];

    // Load the first circle indices into a vector.
    let indices = _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(&circle_offset[0]));

    // Load the thresholds into a vector.
    let m128_threshold = [t as u8; 16];
    let m128_threshold =
        _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(&m128_threshold[0]));

    // Load the center into a vector.
    let m128_center = [base_v as u8; 16];
    let m128_center = _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(&m128_center[0]));
    trace!("m128_center  {}", pi(&m128_center));

    // Perform a single gather to obtain the first 8 pixels from the indices.
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
    let lookup_base = std::mem::transmute::<_, *const i32>(&data[base_offset as usize]);
    let obtained = _mm256_i32gather_epi32(lookup_base, indices, SCALE);

    // after the gather, we end up with the values from our circle like this:
    // v0 0 0 0 v1 0 0 0 v2 0 0 0 v3 0 0 0 | v4 0 0 0 v5 0 0 0 v6 0 0 0 v7
    let mask = _mm256_set_epi64x(
        i64::from_ne_bytes(0x8080808080808080u64.to_ne_bytes()), // 80 for discards
        i64::from_ne_bytes(0x808080800c080400u64.to_ne_bytes()),
        i64::from_ne_bytes(0x8080808080808080u64.to_ne_bytes()), // 80 for discards
        i64::from_ne_bytes(0x808080800c080400u64.to_ne_bytes()),
    );
    // we do this magic shuffle back to collapse that into
    // v0 v1 v2 v3 0000000 | v4 v5 v6 v7
    let first_half = _mm256_shuffle_epi8(obtained, mask);
    // [00, 01, 02, 03, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 04, 05, 06, 07, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00]

    // And, in a second gather, we can get us the remaining 8 indices, exactly the same:
    let indices = _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(
        &circle_offset[SOUTH],
    ));
    let lookup_base = std::mem::transmute::<_, *const i32>(&data[base_offset as usize]);
    let obtained = _mm256_i32gather_epi32(lookup_base, indices, SCALE);

    // Use a different mask here such that we end up with the values in different slots.
    let mask = _mm256_set_epi64x(
        i64::from_ne_bytes(0x808080800c080400u64.to_ne_bytes()),
        i64::from_ne_bytes(0x8080808080808080u64.to_ne_bytes()), // zero out
        i64::from_ne_bytes(0x808080800c080400u64.to_ne_bytes()),
        i64::from_ne_bytes(0x8080808080808080u64.to_ne_bytes()), // zero out
    );
    let second_half = _mm256_shuffle_epi8(obtained, mask);
    // [00, 00, 00, 00, 00, 00, 00, 00, 08, 09, 0A, 0B, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 0C, 0D, 0E, 0F, 00, 00, 00, 00]

    // Then, we can combine the two halves
    let circle_values = _mm256_or_si256(first_half, second_half);
    // [00, 01, 02, 03, 00, 00, 00, 00, 08, 09, 0A, 0B, 00, 00, 00, 00, 04, 05, 06, 07, 00, 00, 00, 00, 0C, 0D, 0E, 0F, 00, 00, 00, 00]

    // Now, we can perform a i32 permutate to end up with everything in the first lane and in order:
    let idx = _mm256_set_epi64x(
        i64::from_ne_bytes(0x0100000001u64.to_ne_bytes()),
        i64::from_ne_bytes(0x0100000001u64.to_ne_bytes()),
        i64::from_ne_bytes(0x0600000002u64.to_ne_bytes()),
        i64::from_ne_bytes(0x0400000000u64.to_ne_bytes()),
    );

    let circle_values_ordered = _mm256_permutevar8x32_epi32(circle_values, idx);
    // [00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 0A, 0B, 0C, 0D, 0E, 0F, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00]

    // Now that we have those points, we can load those back into a vector.
    let p = _mm256_extracti128_si256(circle_values_ordered, 0);
    trace!("Values hex  {}", pi(&p));

    // Definition of the paper is, let a cirle point be p and center of the circle c.
    // darker: p <= c - t
    // similar: c - t < p < c + t
    // brigher: c + t <= p

    // Now, we can calculate the lower and upper bounds.
    let upper_bound = _mm_adds_epu8(m128_center, m128_threshold);
    let lower_bound = _mm_subs_epu8(m128_center, m128_threshold);
    trace!("upper_bound {}", pi(&upper_bound));
    trace!("lower_bound {}", pi(&lower_bound));

    // Perform the compare.
    let is_above = _mm_cmpgt_epu8(p, upper_bound);
    let is_below = _mm_cmpgt_epu8(lower_bound, p);
    trace!("is_above    {}", pi(&is_above));
    trace!("is_below    {}", pi(&is_below));

    const COUNT: usize = 16;
    // Retrieve upper bits of the compare mask, which means everything fits in the top 16 bits.
    let below_bits = _mm_movemask_epi8(is_below);
    let above_bits = _mm_movemask_epi8(is_above);
    trace!("below_bits    {below_bits:?}");
    trace!("above_bits    {above_bits:?}");

    // Use popcount to determine which type of keypoint it is
    let below_count = below_bits.count_ones();
    let above_count = above_bits.count_ones();
    trace!("below_count    {below_count:?}");
    trace!("above_count    {above_count:?}");

    // Then, we only have to operate on a single u32, that has bits set for the pixels that
    // exceed the threshold.
    let used_bits = if below_count > above_count {
        below_bits as u32 | ((below_bits as u32) << 16)
    } else {
        above_bits as u32 | ((above_bits as u32) << 16)
    };

    // Next, we need to figure out if there is a consecutive sequence of above or below
    // in the ring of 16.

    // Working with that ring however, is tricky with the wraparound.
    // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
    //                                                  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
    // We can solve that problem by concatenating the vector again at the end, and iterating
    // over the section that is (16 + consecutive)

    let mut found_consecutive = 0;
    for k in 0..(COUNT + consecutive as usize) {
        if (used_bits & (1 << k)) != 0 {
            found_consecutive += 1;
            // If we found the correct number of consecutive bits, bial out.
            if found_consecutive >= consecutive {
                return Some(FastPoint { x: xx, y });
            }
            // We can also break if we can not possibly reach consecutive before end of iteration
            // Needs a benchmark.
            if (k + (consecutive - found_consecutive) as usize) > (COUNT + consecutive as usize) {
                break;
            }
        } else {
            found_consecutive = 0;
        }
    }

    None
}


pub fn detect(image: &image::GrayImage, t: u8, consecutive: u8) -> Vec<FastPoint> {
    assert!(consecutive >= 9, "number of consecutive pixels needs to exceed 9");

    let height = image.height();
    let width = image.width();
    let t = t as i16;
    // exists range n where all entries different than p - t.
    let mut r = vec![];

    let data = image.as_raw();

    // calculate the circle offsets for the data once.
    let circle_offset = calculate_offsets(width);

    unsafe {
        // Load a vector of thresholds.
        let m128_threshold = [t as u8; 16];
        let m128_threshold =
            _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(&m128_threshold[0]));

        for y in 3..(height - 3) {
            //     15 0 1
            //   14       2
            // 13           3
            // 12    +      4
            // 11           5
            //   10       6
            //     9  8  7

            // Cardinal directions:
            //  n >= 12: 3/4
            //  n >= 9 : 2/4
            // If that is the case, and only then should we do real work.
            // We'll check this in steps of 16 consecutive pixels (center points) in a row.

            const STEP_SIZE: usize = 16;
            let x_chunks = (width - 3 - 3) / STEP_SIZE as u32;
            for x_step in 0..x_chunks {
                let x = 3 + x_step * STEP_SIZE as u32;

                trace!("\n\n");
                // Ok, here we go.
                // __m128i = 16 bytes;
                let base_offset = (y * width + x) as i32;

                // Obtain 16 centers.
                let c = _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(
                    &data[base_offset as usize],
                ));
                trace!("c    : {}", pi(&c));
                trace!("t    : {}", pi(&m128_threshold));

                // Obtain 16 of the cardinal directions.
                let north = _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(
                    &data[(base_offset + circle_offset[NORTH]) as usize],
                ));
                let east = _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(
                    &data[(base_offset + circle_offset[EAST]) as usize],
                ));
                let south = _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(
                    &data[(base_offset + circle_offset[SOUTH]) as usize],
                ));
                let west = _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(
                    &data[(base_offset + circle_offset[WEST]) as usize],
                ));
                trace!("");
                trace!("north: {}", pi(&north));
                trace!("east : {}", pi(&east));
                trace!("south: {}", pi(&south));
                trace!("west:  {}", pi(&west));

                // Ok, great, we have the data on hand, now for each byte, we can
                // Now, we can calculate the lower and upper bounds that we should exceed.
                let upper_bound = _mm_adds_epu8(c, m128_threshold);
                let lower_bound = _mm_subs_epu8(c, m128_threshold);
                trace!("");
                trace!("upbnd: {}", pi(&upper_bound));
                trace!("lrbnd: {}", pi(&lower_bound));

                // Perform the compare to determine if the point is beyond the thresholds.
                // These masks return 0xFF if true, x00 if false.
                // _mm_cmpgt_epi8: dst[i+7:i] := ( a[i+7:i] > b[i+7:i] ) ? 0xFF : 0
                let north_above = _mm_cmpgt_epu8(north, upper_bound);
                let east_above = _mm_cmpgt_epu8(east, upper_bound);
                let south_above = _mm_cmpgt_epu8(south, upper_bound);
                let west_above = _mm_cmpgt_epu8(west, upper_bound);
                trace!("");
                trace!("nt_ab: {}", pi(&north_above));
                trace!("ea_ab: {}", pi(&east_above));
                trace!("st_ab: {}", pi(&south_above));
                trace!("we_ab: {}", pi(&west_above));

                let north_below = _mm_cmpgt_epu8(lower_bound, north);
                let east_below = _mm_cmpgt_epu8(lower_bound, east);
                let south_below = _mm_cmpgt_epu8(lower_bound, south);
                let west_below = _mm_cmpgt_epu8(lower_bound, west);
                trace!("");
                trace!("nt_bl: {}", pi(&north_below));
                trace!("ea_bl: {}", pi(&east_below));
                trace!("st_bl: {}", pi(&south_below));
                trace!("we_bl: {}", pi(&west_below));

                trace!("");

                // Next, depending on the number of consecutive points, we perform some masking to
                // determine if the point is a potential keypoint.
                // Yes, there is a branch in the loop here, but it doesn't seem to have a
                // tremenduous amount of impact.
                let check_mask = if consecutive < 12 && consecutive >= 9 {
                    // Now, we need a way to determine 2 out of 4.
                    //               && south && west
                    // north &&               && west
                    // north && east &&       &&
                    //       && east && south &&

                    let above_0 = _mm_and_si128(south_above, west_above);
                    let above_1 = _mm_and_si128(north_above, west_above);
                    let above_2 = _mm_and_si128(north_above, east_above);
                    let above_3 = _mm_and_si128(east_above, south_above);
                    // Finally, or those because any of them makes it a potential keypoint.
                    let above_2_found = _mm_or_si128(
                        _mm_or_si128(above_0, above_1),
                        _mm_or_si128(above_2, above_3),
                    );
                    trace!("2 gtf: {}", pi(&above_2_found));

                    // And the same for below.
                    let below_0 = _mm_and_si128(south_below, west_below);
                    let below_1 = _mm_and_si128(north_below, west_below);
                    let below_2 = _mm_and_si128(north_below, east_below);
                    let below_3 = _mm_and_si128(east_below, south_below);
                    let below_2_found = _mm_or_si128(
                        _mm_or_si128(below_0, below_1),
                        _mm_or_si128(below_2, below_3),
                    );
                    trace!("2 ltf: {}", pi(&below_2_found));

                    let found_3 = _mm_or_si128(above_2_found, below_2_found);

                    let mut mask = [0u8; STEP_SIZE];
                    _mm_storeu_si128(
                        std::mem::transmute::<_, *mut __m128i>(&mut mask[0]),
                        found_3,
                    );
                    mask
                } else if consecutive >= 12 {
                    // Now, we need a way to determine 3 out of 4.
                    //          east && south && west
                    // north &&         south && west
                    // north && east &&       && west
                    // north && east && south &&

                    let above_0 = _mm_and_si128(_mm_and_si128(east_above, south_above), west_above);
                    let above_1 =
                        _mm_and_si128(_mm_and_si128(north_above, south_above), west_above);
                    let above_2 = _mm_and_si128(_mm_and_si128(north_above, east_above), west_above);
                    let above_3 =
                        _mm_and_si128(_mm_and_si128(north_above, east_above), south_above);
                    let above_3_found = _mm_or_si128(
                        _mm_or_si128(above_0, above_1),
                        _mm_or_si128(above_2, above_3),
                    );
                    trace!("3 gtf: {}", pi(&above_3_found));

                    // And the same for below.
                    let below_0 = _mm_and_si128(_mm_and_si128(east_below, south_below), west_below);
                    let below_1 =
                        _mm_and_si128(_mm_and_si128(north_below, south_below), west_below);
                    let below_2 = _mm_and_si128(_mm_and_si128(north_below, east_below), west_below);
                    let below_3 =
                        _mm_and_si128(_mm_and_si128(north_below, east_below), south_below);
                    let below_3_found = _mm_or_si128(
                        _mm_or_si128(below_0, below_1),
                        _mm_or_si128(below_2, below_3),
                    );
                    trace!("3 ltf: {}", pi(&below_3_found));

                    let found_3 = _mm_or_si128(above_3_found, below_3_found);

                    let mut mask = [0u8; STEP_SIZE];
                    _mm_storeu_si128(
                        std::mem::transmute::<_, *mut __m128i>(&mut mask[0]),
                        found_3,
                    );
                    mask
                } else {
                    [0xFFu8; STEP_SIZE]
                };

                // Finally, iterate over the potential candidates and determine if they were really
                // a keypoint using the more extensive checks.
                for xx in x..(x + STEP_SIZE as u32) {
                    if check_mask[(xx - x) as usize] == 0 {
                        continue;
                    }
                    if let Some(keypoint) = determine_keypoint(
                        data,
                        &circle_offset,
                        width,
                        (xx, y),
                        t as u8,
                        consecutive,
                    ) {
                        r.push(keypoint);
                    }
                }
            }

            // Clean up the rest of the row that didn't fit in chunks of 16.
            for x_step in ((width - 3 - 3) / STEP_SIZE as u32) * (STEP_SIZE as u32)..(width - 3 - 3)
            {
                // for x in (width - 16 - 3)..(width -3){
                let x = x_step + 3;
                if let Some(keypoint) =
                    determine_keypoint(data, &circle_offset, width, (x, y), t as u8, consecutive)
                {
                    r.push(keypoint);
                }
            }
        }
    }
    r
}

pub fn detector(img: &image::GrayImage, config: &FastConfig) -> Vec<FastPoint> {
    detect(img, config.threshold, config.count)
}

#[cfg(test)]
mod test {
    use super::*;
    use image::Luma;

    fn create_sample_image(center: u8, circle_values: &[u8]) -> image::GrayImage {
        let w = 128;
        let h = 128;
        let mut z = image::GrayImage::new(w, h);
        z.put_pixel(w / 2, h / 2, Luma::<u8>([center]));
        assert_eq!(circle_values.len(), 16);
        for (i, v) in circle_values.iter().enumerate() {
            let offset = circle()[i];
            z.put_pixel(
                ((w / 2) as i32 + offset.0) as u32,
                ((h / 2) as i32 + offset.1) as u32,
                Luma::<u8>([*v]),
            );
        }
        z
    }

    #[test]
    fn test_47_115_hand() {
        /*
            Determine keypoint at (64, 64)
            m128_center  [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
            Values dec  [37, 37, 39, 39, 37, 42, 43, 16, 14, 13, 15, 16, 15, 38, 37, 38]
            Values hex  [25, 25, 27, 27, 25, 2A, 2B, 10, 0E, 0D, 0F, 10, 0F, 26, 25, 26]
            upper_bound [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21]
            lower_bound [01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01]
            is_above    [FF, FF, FF, FF, FF, FF, FF, 00, 00, 00, 00, 00, 00, FF, FF, FF]
            is_below    [00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00]
            above_u8    [255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255]
            below_u8    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        */
        let img = create_sample_image(
            17,
            &[
                // N              E                S              W
                37, 37, 39, 39, 37, 42, 43, 16, 14, 13, 15, 16, 15, 38, 37, 38,
            ],
        );
        let _ = img
            .save("/tmp/test_47_115_hand.png")
            .expect("should be able to write image");
        let threshold = 16;
        let count_minimum = 9;

        // let height = img.height();
        let width = img.width();
        let circle_offset = calculate_offsets(width);
        let z = unsafe {
            determine_keypoint(
                &img.as_raw(),
                &circle_offset,
                width,
                (img.width() / 2, img.height() / 2),
                threshold,
                count_minimum,
            )
        };
        assert!(z.is_some());

        let detected = detect(&img, 16, 9);
        assert_eq!(
            detected.contains(&FastPoint {
                x: img.width() / 2,
                y: img.height() / 2
            }),
            true
        );
    }

    #[test]
    fn test_combine_gathers() {
        unsafe {
            let data = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let mut indices = [0u32; 16];
            for k in 0..16 {
                indices[k] = k as u32;
            }

            let indices = _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(&indices[0]));
            // us the first 8 indices.
            const SCALE: i32 = 1;
            let lookup_base = std::mem::transmute::<_, *const i32>(&data[0]);
            let obtained = _mm256_i32gather_epi32(lookup_base, indices, SCALE);

            // after the gather, we end up with the values from our circle like this:
            // v0 0 0 0 v1 0 0 0 v2 0 0 0 v3 0 0 0 | v4 0 0 0 v5 0 0 0 v6 0 0 0 v7
            let mask = _mm256_set_epi64x(
                i64::from_ne_bytes(0x8080808080808080u64.to_ne_bytes()), // discarded anyway
                // on zero'th byte, we want the 0 index, second byte, index 4, third; 8th...
                i64::from_ne_bytes(0x808080800c080400u64.to_ne_bytes()),
                i64::from_ne_bytes(0x8080808080808080u64.to_ne_bytes()), // discarded anyway
                i64::from_ne_bytes(0x808080800c080400u64.to_ne_bytes()),
            );
            // we do this magic shuffle back to collapse that into
            // v0 v1 v2 v3 0000000 | v4 v5 v6 v7
            let first_half = _mm256_shuffle_epi8(obtained, mask);

            let lookup_base = std::mem::transmute::<_, *const i32>(&data[8]);
            let obtained_second = _mm256_i32gather_epi32(lookup_base, indices, SCALE);

            let mask = _mm256_set_epi64x(
                i64::from_ne_bytes(0x808080800c080400u64.to_ne_bytes()),
                i64::from_ne_bytes(0x8080808080808080u64.to_ne_bytes()), // zero out
                i64::from_ne_bytes(0x808080800c080400u64.to_ne_bytes()),
                i64::from_ne_bytes(0x8080808080808080u64.to_ne_bytes()), // zero out
            );
            let second_half = _mm256_shuffle_epi8(obtained_second, mask);

            // Or that, such that we end up with
            // v0 v1 v2 v3 0 0 0 0 v8 v9 v10 v11  0 0 0 0 v4 v5 v6 v7 0 0 0 0 v12 v13 v14 v15 0 0 0 0
            let combined = _mm256_or_si256(first_half, second_half);

            println!("first_half:  {}", pl(&first_half));
            println!("second_half: {}", pl(&second_half));
            println!("             {}", pl(&combined));

            // Now, we're a single permutate away from getting what we want.

            // v0 v1 v2 v3 0 0 0 0 v8 v9 v10 v11  0 0 0 0 v4 v5 v6 v7 0 0 0 0 v12 v13 v14 v15 0 0 0 0
            // Needs to become
            // v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 0000000
            let idx = _mm256_set_epi64x(
                i64::from_ne_bytes(0x0100000001u64.to_ne_bytes()),
                i64::from_ne_bytes(0x0100000001u64.to_ne_bytes()),
                i64::from_ne_bytes(0x0600000002u64.to_ne_bytes()),
                i64::from_ne_bytes(0x0400000000u64.to_ne_bytes()),
            );
            // __m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx)
            let tmp = _mm256_permutevar8x32_epi32(combined, idx);
            println!("             {}", pl(&tmp));
        }
    }
}
