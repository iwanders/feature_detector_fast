//!
//! This is a highly optimised implementation of the FAST feature detector.
//! It makes heavy use of the AVX2 instruction set to achieve the highest possible throughput.
//!
//!
//! # Algorithm:
//!   Extremely concise version of the algorithm.
//!   - Compare points on a circle against the center point, circle has 16 points.
//!   - Check if points on the circle exceed center point by a threshold.
//!   Definition of the paper is, let a cirle point be p and center of the circle c.
//!     - darker: `p <= c - t`
//!     - similar: `c - t < p < c + t`
//!     - brigher: `c + t <= p`
//!   - If there are more than n consecutive pixels on the circle that are darker, the center point is a feature.
//!   - If there are more than n consecutive pixels on the circle that are lighter, the center point is a feature.
//!   - If non maximum suppression is enabled, calculate that for each candidate feature, they only become features if they are the strongest feature compared to their 8 neighbours.
//!   
//!
//!# Overall approach is:
//!   - Iterate through each row.
//!     - Iterate through the row in blocks of 16 pixels. The center pixels.
//!     - For each center, determine whether the cardinal directions exceed the threshold.
//!       For n >= 12: 3/4 need to match.
//!       For n > 9 && n < 12: 2/4 need to match.
//!       If these match, set a bit to mark it as a potential.
//!     - Iterate throught potentials and perform the thorough check.
//!       - Thorough check uses two gather operations to retrieve the entire circle's pixels
//!       - Wrangle these points (conveniently 16 of them) into a single 128 bit vector.
//!       - Determine if points exceed the threshold above or below the center point.
//!       - Reduce the data to a single integer, each bit representing whether the bound is exceeded.
//!       - Use popcnt to determine if it is a positive or negative keypoint
//!       - Iterate throught the correct integer, checking if the correct number of consecutive
//!         value exceeds has been found.
//!
//! # Tests
//! Tests ran on my system (i7-4770TE, from 2014) against a 1920 x 1080 grayscale image from a
//! computer game.
//!
//! #### Results without non-maximum supression:
//!   - OpenCV takes 18'ish milliseconds to run with a threshold of 16, 9/16 consecutive, no nonmax supression. This finds 23184 keypoints.
//!   - This implementation takes 10'ish milliseconds, with the same parameters. And finds the same 23184 keypoints.
//! #### Results with non-maximum supression:
//!   - OpenCV takes 31'ish milliseconds to run with a threshold of 16, 9/16 consecutive, nonmax supression using maximum 't' for which it is a keypoint. This finds 7646 keypoints.
//!   - This implementation takes 14'ish milliseconds, with the same parameters. And finds the same 7646 keypoints.
//!
//! # Remarks
//!   - The current non maximum supression score function is the maximum 't' for which a feature would still be a feature.
//!     According to the paper this score function is often very similar between pixels in an image.
//!

use crate::{FastConfig, FastPoint};
use std::arch::x86_64::*;

// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=MMX&avxnewtechs=AVX,AVX2&ssetechs=SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2

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

const NONMAX_DISABLED: u8 = 0;
const NONMAX_MAX_THRESHOLD: u8 = NONMAX_DISABLED + 1;
const NONMAX_SUM_ABSOLUTE: u8 = NONMAX_MAX_THRESHOLD + 1;

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

// This inline keyword actually makes a difference.
#[inline(always)]
/// Determine whether there's a keypoint at the provided location.
unsafe fn determine_keypoint<const NONMAX: u8>(
    data: &[u8],
    circle_offset: &CircleOffsets,
    width: u32,
    p: (u32, u32),
    t: u8,
    consecutive: u8,
    score: Option<&mut u16>,
) -> bool {
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
                if NONMAX == NONMAX_DISABLED {
                    return true;
                } else if NONMAX == NONMAX_MAX_THRESHOLD {
                    // Need to calculate the score.
                    *score.unwrap() = keypoint_score_max_threshold(base_v, p, consecutive);
                    return true;
                } else if NONMAX == NONMAX_SUM_ABSOLUTE {
                    // Need to calculate the score.
                    // keypoint_score_sum_abs_difference(pixels: __m128i, centers: __m128i, is_above: __m128i, is_below: __m128i, threshold: __m128i)
                    *score.unwrap() = keypoint_score_sum_abs_difference(
                        p,
                        m128_center,
                        is_above,
                        is_below,
                        m128_threshold,
                    );
                    return true;
                }
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

    false
}

// It would be really nice to have feature(adt_const_params) here; https://github.com/rust-lang/rust/issues/95174

pub fn detect<const NONMAX: u8>(
    image: &image::GrayImage,
    t: u8,
    consecutive: u8,
) -> Vec<FastPoint> {
    assert!(
        consecutive >= 9,
        "number of consecutive pixels needs to exceed 9"
    );

    let height = image.height();
    let width = image.width();

    // Nonmax suppression works with blocks of 3x3, instead of inserting the keypoints immediately
    // we will need to keep track of them, until have calculated the neighbours and can properly
    // evaluate their insertion criteria.
    // All nonmax related variables are prepended nonmax_, because it is a const parameter, the
    // compiler will hopefully optimise them out in the nonmax flavour.
    //
    // We will keep a moving window of three rows;
    // row y-2
    // row y-1
    // row y-0, current
    // At the end of each row, we will iterate through y-1, and finalise keypoints based on whether
    // the neighbour scores are lower / higher / better, etc.
    let mut nonmax_pending_insertions: Vec<u16> = vec![0; width as usize * 3];
    let (mut nonmax_first, nonmax_second) = nonmax_pending_insertions.split_at_mut(width as usize);
    let (mut nonmax_second, mut nonmax_third) = nonmax_second.split_at_mut(width as usize);
    // let nonmax_rows = [nonmax_first, nonmax_second, nonmax_third];

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

            // Create the nonmax rows at the appropriate indices, not elegant :(
            let mut nonmax_rows = [&mut nonmax_first, &mut nonmax_second, &mut nonmax_third];
            nonmax_rows.rotate_left(y as usize % 3);
            let (nonmax_y_2, nonmax_y_1_0) = nonmax_rows.split_at_mut(1);
            let nonmax_y_2 = &mut nonmax_y_2[0];
            let (nonmax_y_1, nonmax_y_0) = nonmax_y_1_0.split_at_mut(1);
            let nonmax_y_1 = &mut nonmax_y_1[0];
            let nonmax_y_0 = &mut nonmax_y_0[0];
            // start of the row, clear the current;
            nonmax_y_0.fill(0);

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
                    let mut nonmax_score: u16 = 0;
                    let nonmax_optional_score = if NONMAX != NONMAX_DISABLED {
                        Some(&mut nonmax_score)
                    } else {
                        None
                    };
                    if determine_keypoint::<NONMAX>(
                        data,
                        &circle_offset,
                        width,
                        (xx, y),
                        t as u8,
                        consecutive,
                        nonmax_optional_score,
                    ) {
                        if NONMAX == NONMAX_DISABLED {
                            r.push(FastPoint { x: xx, y: y });
                        } else {
                            nonmax_y_0[xx as usize] = nonmax_score;
                        }
                    }
                }
            }

            // Clean up the rest of the row that didn't fit in chunks of 16.
            for x_step in ((width - 3 - 3) / STEP_SIZE as u32) * (STEP_SIZE as u32)..(width - 3 - 3)
            {
                // for x in (width - 16 - 3)..(width -3){
                let x = x_step + 3;

                let mut nonmax_score: u16 = 0;
                let nonmax_optional_score = if NONMAX != NONMAX_DISABLED {
                    Some(&mut nonmax_score)
                } else {
                    None
                };
                if determine_keypoint::<NONMAX>(
                    data,
                    &circle_offset,
                    width,
                    (x, y),
                    t as u8,
                    consecutive,
                    nonmax_optional_score,
                ) {
                    if NONMAX == NONMAX_DISABLED {
                        r.push(FastPoint { x: x, y: y });
                    } else {
                        nonmax_y_0[x as usize] = nonmax_score;
                    }
                }
            }

            // At the end of the row, if we are doing nonmax, iterate through the appropriate row, and evaluate.
            if NONMAX != NONMAX_DISABLED {
                if y == 4 {
                    continue; // Quirk from OpenCV?
                }
                let above = &nonmax_y_2;
                let center = &nonmax_y_1;
                let below = &nonmax_y_0;
                for x in 3..(width as usize - 3) {
                    // check if we even have a point here.
                    if center[x] == 0 {
                        continue;
                    }

                    // our score shorthand
                    let s = center[x];
                    let exceed_above = s > above[x - 1] && s > above[x - 0] && s > above[x + 1];
                    let exceed_cntr = s > center[x - 1] && s > center[x + 1];
                    let exceed_below = s > below[x - 1] && s > below[x - 0] && s > below[x + 1];

                    if exceed_above && exceed_cntr && exceed_below {
                        r.push(FastPoint {
                            x: x as u32,
                            y: y - 1,
                        });
                    }
                }
            }
        } // row iteration end.
    }
    r
}

/// This is a vectorized equivalent to [`crate::opencv_compat::non_max_suppression_opencv_score`].
pub fn keypoint_score_max_threshold(base_v: u8, pixels: __m128i, consecutive: u8) -> u16 {
    unsafe {
        // duplicate that to both lanes.
        let pixels = _mm256_set_m128i(_mm_set1_epi64x(0), pixels); // combine two 128 into 256

        let centers = _mm256_set1_epi16(base_v as i16);

        // Perform a permutate to go from:
        // v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 0 0 0 0 0 0 0 0
        // to
        // v0 v1 v2 v3 v4 v5 v6 v7 0 0 0 0 0 0 0 0 ... v8 v9 v10 v11 v12 v13 v14 v15 0 0 0 0 0 0 0 0
        //
        // Now, we can perform a i32 permutate to end up with everything in the first lane and in order:
        let idx = _mm256_set_epi64x(
            i64::from_ne_bytes(0x07_00000007u64.to_ne_bytes()),
            i64::from_ne_bytes(0x03_00000002u64.to_ne_bytes()),
            i64::from_ne_bytes(0x07_00000007u64.to_ne_bytes()),
            i64::from_ne_bytes(0x01_00000000u64.to_ne_bytes()),
        );

        let pixels_two_lanes = _mm256_permutevar8x32_epi32(pixels, idx);
        // So now we have 8 pixels in the left, and 8 pixels in the right lane.

        // Next, we unpack that from:
        // v0 v1 v2 v3 v4 v5 v6 v7 0 0 0 0 0 0 0 0
        // to
        // v0 0  v1 0 v2 0 v3 0  v4 0 v5 0 v6 0 v7 0
        let zeros = _mm256_set1_epi64x(0);
        let as_i16 = _mm256_unpacklo_epi8(pixels_two_lanes, zeros);
        // So now we have an mm256 vector with u16 / i16 values.

        // Calculate the difference vector, but offset centers by 512 to stay all positive.
        let difference_vector_plus_512 =
            _mm256_sub_epi16(_mm256_add_epi16(centers, _mm256_set1_epi16(512)), as_i16);
        // Make that mutable, we'll rotate it as we go along.
        let mut difference_vector = difference_vector_plus_512;

        // Next up, create a mask of 'consecutive' long.
        let mut consec_mask = [0u16; 16];
        for i in 0..consecutive {
            consec_mask[i as usize] = 0xffff;
        }
        let consec_mask =
            _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(&consec_mask[0]));
        let and_not_mask = _mm256_andnot_si256(consec_mask, _mm256_set1_epi8(-1));

        // Allocate vectors to hold the min and max value found in each 'consecutive' block.
        let mut min_values = [0x0u16; 16];
        let mut max_values = [0xFFFFu16; 16];

        // Iterate over all 16 possible rotations.
        for k in 0..16 {
            // Calculate the minimum in this sequence.
            {
                let masked_seq = _mm256_and_si256(difference_vector, consec_mask);
                let masked_rem_max = _mm256_or_si256(masked_seq, and_not_mask);

                // Retrieve the minimum u16 in the vector and store that.
                let calculated_min = _mm256_minpos_epu16(masked_rem_max);
                min_values[k] = calculated_min;
            }

            // Calculate the maximum in this sequence.
            {
                // We only have min _mm256_minpos_epu16. so to do the max, we'll need to subtract diff
                // from a large enough value.
                let difference_vector_from_top =
                    _mm256_sub_epi16(_mm256_set1_epi16(-1), difference_vector);
                let masked_seq = _mm256_and_si256(difference_vector_from_top, consec_mask);
                let masked_rem_max = _mm256_or_si256(masked_seq, and_not_mask);

                // Retrieve the minimum u16 in the vector and store that.
                let calculated_max = _mm256_minpos_epu16(masked_rem_max);
                max_values[k] = calculated_max;
            }

            // Then, rotate the vector.
            difference_vector = _mm256_rotate_across_2(difference_vector);
        }

        // Now, we need a max operation.
        let min_values_vector =
            _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(&min_values[0]));
        let min_values_from_top = _mm256_sub_epi16(_mm256_set1_epi16(1024), min_values_vector);
        let lowest_min_values = _mm256_minpos_epu16(min_values_from_top);
        // And finally translate this back to a signed value.
        let extreme_highest = 1024 - lowest_min_values as i16 - 512;

        // And our min operation.
        let max_values_vector =
            _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(&max_values[0]));
        let max_values_from_top = _mm256_sub_epi16(_mm256_set1_epi16(1024), max_values_vector);
        let lowest_max_values = _mm256_minpos_epu16(max_values_from_top);
        // And finally translate this back to a signed value.
        let extreme_lowest = (lowest_max_values - (1024 + 512 + 1)) as i16;

        // Take the absolute minimum of both to determine the max 't' for which this is a point.
        let res = extreme_highest.abs().min(extreme_lowest.abs()) as u16;

        res
    }
}

pub fn keypoint_score_sum_abs_difference(
    pixels: __m128i,
    centers: __m128i,
    is_above: __m128i,
    is_below: __m128i,
    threshold: __m128i,
) -> u16 {
    unsafe {
        // trace!("centers       {}", pi(&centers));
        // trace!("pixels        {}", pi(&pixels));
        // trace!("threshold     {}", pi(&threshold));
        let values_bright = _mm_and_si128(
            _mm_subs_epu8(_mm_subs_epu8(centers, pixels), threshold),
            is_below,
        );
        let values_dark = _mm_and_si128(
            _mm_subs_epu8(_mm_subs_epu8(pixels, centers), threshold),
            is_above,
        );
        // trace!("is_above      {}", pi(&is_above));
        // trace!("is_below      {}", pi(&is_below));
        // trace!("values_bright {}", pi(&values_bright));
        // trace!("values_dark   {}", pi(&values_dark));
        let bright = _mm_sum_epu8(values_bright);
        let dark = _mm_sum_epu8(values_dark);
        bright.max(dark) as u16
    }
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
/// Calculate the minimum u16 of a m256 vector.
unsafe fn _mm256_minpos_epu16(v: __m256i) -> u16 {
    let lower = _mm256_extracti128_si256(v, 0);
    let upper = _mm256_extracti128_si256(v, 1);
    let lower_lowest = _mm_minpos_epu16(lower);
    let upper_lowest = _mm_minpos_epu16(upper);
    let lower_lowest = _mm_extract_epi16(lower_lowest, 0);
    let upper_lowest = _mm_extract_epi16(upper_lowest, 0);
    lower_lowest.min(upper_lowest) as u16
}

/// Rotate an m256 by two bytes, across lanes.
unsafe fn _mm256_rotate_across_2(difference_vector: __m256i) -> __m256i {
    // __m256i _mm256_alignr_epi8 (__m256i a, __m256i b, const int imm8)
    let rotated = _mm256_alignr_epi8(difference_vector, difference_vector, 2);
    // println!("r:       {}", pl(&rotated));

    // Gah, that's per lane... so we need to swap the top of both lanes.
    // __m256i _mm256_insert_epi16 (__m256i a, __int16 i, const int index)

    let rotated_swap = _mm256_permute2x128_si256(rotated, rotated, 1);
    // println!("rswap    {}", pl(&rotated_swap));

    // Cool, with the lanes swapped, we can blend the two lanes;
    let mask = _mm256_set_epi64x(
        i64::from_ne_bytes(0xFFFF0000_00000000u64.to_ne_bytes()),
        i64::from_ne_bytes(0x00000000_00000000u64.to_ne_bytes()),
        i64::from_ne_bytes(0xFFFF0000_00000000u64.to_ne_bytes()),
        i64::from_ne_bytes(0x00000000_00000000u64.to_ne_bytes()),
    );
    let rotated = _mm256_blendv_epi8(rotated, rotated_swap, mask);
    // println!("rot      {}", pl(&rotated));
    rotated
}

/// Horizontally sum a vector of u8's.
// https://stackoverflow.com/a/36998778
unsafe fn _mm_sum_epu8(v: __m128i) -> u32 {
    let vsum = _mm_sad_epu8(v, _mm_setzero_si128());
    u32::from_ne_bytes((_mm_cvtsi128_si32(vsum) + _mm_extract_epi16(vsum, 4)).to_ne_bytes())
}

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
    format!(
        "{} | {}",
        format!("{:02X?}", &v[0..16]),
        format!("{:02X?}", &v[16..])
    )
}

pub fn detector(img: &image::GrayImage, config: &FastConfig) -> Vec<FastPoint> {
    match config.non_maximal_supression {
        crate::NonMaximalSuppression::Off => {
            detect::<NONMAX_DISABLED>(img, config.threshold, config.count)
        }
        crate::NonMaximalSuppression::MaxThreshold => {
            detect::<NONMAX_MAX_THRESHOLD>(img, config.threshold, config.count)
        }
        crate::NonMaximalSuppression::SumAbsolute => {
            detect::<NONMAX_SUM_ABSOLUTE>(img, config.threshold, config.count)
        }
    }
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

    fn create_random_image(seed: u64) -> image::GrayImage {
        use rand_xoshiro::rand_core::{RngCore, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let center = rng.next_u32() as u8;
        let mut circle_values = [0u8; 16];
        for v in circle_values.iter_mut() {
            *v = rng.next_u32() as u8;
        }
        create_sample_image(center, &circle_values)
    }

    fn opencv_nonmax_test_wrapper(image: &image::GrayImage, (x, y): (u32, u32)) -> u16 {
        unsafe {
            // Definition of the paper is, let a cirle point be p and center of the circle c.
            //     darker: p <= c - t
            //     similar: c - t < p < c + t
            //     brigher: c + t <= p
            //
            let base_v = image.get_pixel(x, y)[0];

            let mut points = [0 as u8; 16];
            let offsets = circle();
            for i in 0..offsets.len() {
                let pos = circle()[i];
                let p = image.get_pixel((x as i32 + pos.0) as u32, (y as i32 + pos.1) as u32)[0];
                points[i] = p;
            }

            // First, collect the pixels into a vector.
            let pixels = _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(&points[0]));
            keypoint_score_max_threshold(base_v, pixels, 9)
        }
    }

    #[test]
    fn test_47_115_score_calc() {
        // let center = 10;
        let center = 17;
        let img = create_sample_image(
            center,
            &[
                // N              E                S              W
                37, 37, 39, 39, 37, 42, 43, 16, 14, 13, 15, 16, 15, 38, 37,
                38,
                // 1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ],
        );
        let x = img.width() / 2;
        let y = img.height() / 2;
        let score = crate::opencv_compat::non_max_suppression_opencv_score(&img, (x, y));
        assert_eq!(opencv_nonmax_test_wrapper(&img, (x, y)), 20);
        assert_eq!(score, 20);

        for i in 0..20000 {
            println!("i: {i:?}");
            let img = create_random_image(i);
            let x = img.width() / 2;
            let y = img.height() / 2;
            let score = crate::opencv_compat::non_max_suppression_opencv_score(&img, (x, y));
            assert_eq!(opencv_nonmax_test_wrapper(&img, (x, y)), score);
        }

        // Cool, now we have our vector, we just need to... do things, extremely fast.
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

        let x = img.width() / 2;
        let y = img.height() / 2;

        // let height = img.height();
        let width = img.width();
        let circle_offset = calculate_offsets(width);

        let z = unsafe {
            determine_keypoint::<NONMAX_DISABLED>(
                &img.as_raw(),
                &circle_offset,
                width,
                (x, y),
                threshold,
                count_minimum,
                None,
            )
        };
        assert!(z);

        let detected = detect::<NONMAX_DISABLED>(&img, 16, 9);
        assert_eq!(detected.contains(&FastPoint { x, y }), true);

        // Check the score function.
        let score = crate::opencv_compat::non_max_suppression_opencv_score(&img, (x, y));
        assert_eq!(score, 20);

        let mut calculated_score = 0u16;
        let z = unsafe {
            determine_keypoint::<NONMAX_MAX_THRESHOLD>(
                &img.as_raw(),
                &circle_offset,
                width,
                (x, y),
                threshold,
                count_minimum,
                Some(&mut calculated_score),
            )
        };
        assert!(z);
        assert_eq!(calculated_score, score);
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

    #[test]
    fn test_mm256_minpos_epu16() {
        use rand_xoshiro::rand_core::{RngCore, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        unsafe {
            for _i in 0..100000 {
                let mut indices = [0u16; 16];
                for k in 0..16 {
                    indices[k] = rng.next_u32() as u16;
                }

                let indices_vector =
                    _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(&indices[0]));
                let min_simd = _mm256_minpos_epu16(indices_vector);
                assert_eq!(min_simd, *indices.iter().min().unwrap());
            }
        }
    }

    #[test]
    fn test_mm_sum_epu8() {
        use rand_xoshiro::rand_core::{RngCore, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        unsafe {
            for _i in 0..100000 {
                let mut indices = [0u8; 16];
                for k in 0..16 {
                    indices[k] = rng.next_u32() as u8;
                }

                let indices_vector =
                    _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(&indices[0]));
                let sum_simd = _mm_sum_epu8(indices_vector);
                assert_eq!(sum_simd, indices.iter().map(|x| *x as u32).sum::<u32>());
            }
        }
    }

    #[test]
    fn test_mm256_rotate_across_2() {
        use rand_xoshiro::rand_core::{RngCore, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        unsafe {
            for _i in 0..100000 {
                let mut indices = [0u16; 16];
                for k in 0..16 {
                    indices[k] = rng.next_u32() as u16;
                }

                let indices_vector =
                    _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(&indices[0]));
                let rotated_indices = _mm256_rotate_across_2(indices_vector);
                let mut back_to_thing = [0u16; 16];
                _mm256_storeu_si256(
                    std::mem::transmute::<_, *mut __m256i>(&mut back_to_thing[0]),
                    rotated_indices,
                );

                // Rotate with rust;
                indices.rotate_left(1);

                assert_eq!(indices, back_to_thing);
            }
        }
    }

    #[test]
    fn test_score_function_3() {
        // The sum of the absolute difference between the pixels in the contiguous arc and the centre pixel.
        // p is circle, c is center
        // \sigma_{x\in Sbright}|I_p - I_c| - t
        // \sigma_{x\in Sdark}|I_c - I_p| - t
        // - darker: `p <= c - t`
        // - similar: `c - t < p < c + t`
        // - brigher: `c + t <= p`
        // Both |I_p - I_c| and |I_c - I_p| are always positive, since they must exceed the threshold.
        // So all we need to do is take the differences, subtract t, mask with cmp, and finally sum it.
        use rand_xoshiro::rand_core::{RngCore, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        unsafe {
            for _i in 0..10000000 {
                let mut circle = [0u8; 16];
                for k in 0..16 {
                    circle[k] = rng.next_u32() as u8;
                }
                let base_v = rng.next_u32() as u8;
                let t = rng.next_u32() as u8;

                let score_without_simd =
                    crate::opencv_compat::score_non_max_supression_max_abs_sum(base_v, &circle, t);

                let m128_center = _mm_set1_epi8(i8::from_ne_bytes(base_v.to_ne_bytes()));
                let m128_threshold = _mm_set1_epi8(i8::from_ne_bytes(t.to_ne_bytes()));
                let p = _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(&circle[0]));

                // Now, we can calculate the lower and upper bounds.
                let upper_bound = _mm_adds_epu8(m128_center, m128_threshold);
                let lower_bound = _mm_subs_epu8(m128_center, m128_threshold);
                // trace!("upper_bound {}", pi(&upper_bound));
                // trace!("lower_bound {}", pi(&lower_bound));

                // Perform the compare.
                let is_above = _mm_cmpgt_epu8(p, upper_bound);
                let is_below = _mm_cmpgt_epu8(lower_bound, p);
                // trace!("is_above    {}", pi(&is_above));
                // trace!("is_below    {}", pi(&is_below));
                let with_simd = keypoint_score_sum_abs_difference(
                    p,
                    m128_center,
                    is_above,
                    is_below,
                    m128_threshold,
                );
                assert_eq!(score_without_simd, with_simd);
            }
        }
    }
}
