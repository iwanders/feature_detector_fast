use crate::fast::FastPoint;

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
    #[allow(dead_code)]
    unsafe fn pi(input: &__m128i) -> String {
        let v: [u8; 16] = [0; 16];
        _mm_storeu_si128(v.as_ptr() as *mut _, *input);
        format!("{:02X?}", v)
    }
    // Print long simd type
    #[allow(dead_code)]
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

    pub type CircleOffsets = [i32; 16];
    pub fn calculate_offsets(width: u32) -> CircleOffsets {
        let mut circle_offset = [0i32; 16];
        for (i, (x, y)) in circle().iter().enumerate() {
            circle_offset[i] = *y * width as i32 + *x;
        }
        circle_offset
    }
    // ah, this is a signed comparison...
    // https://stackoverflow.com/a/24234695
    /*
    _mm_cmpgt_epu8(a, b) = _mm_cmpgt_epi8(
        _mm_xor_epi8(a, _mm_set1_epi8(-128)),  // range-shift to unsigned
        _mm_xor_epi8(b, _mm_set1_epi8(-128)))
    */
    unsafe fn _mm_cmpgt_epu8(a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpgt_epi8(
            _mm_xor_si128(a, _mm_set1_epi8(-128)), // range-shift to unsigned
            _mm_xor_si128(b, _mm_set1_epi8(-128)),
        )
    }

    unsafe fn _mm256_cmpgt_epu8(a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpgt_epi8(
            _mm256_xor_si256(a, _mm256_set1_epi8(-128)), // range-shift to unsigned
            _mm256_xor_si256(b, _mm256_set1_epi8(-128)),
        )
    }

    #[inline]
    pub unsafe fn determine_keypoint(
        data: &[u8],
        circle_offset: &CircleOffsets,
        width: u32,
        p: (u32, u32),
        t: u8,
        consecutive: u8,
    ) -> Option<FastPoint> {
        trace!("\n\nDetermine keypoint at {p:?}");
        let indices =
            _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(&circle_offset[0]));
        let m128_threshold = [t as u8; 16];
        let m128_threshold =
            _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(&m128_threshold[0]));

        let xx = p.0;
        let y = p.1;
        let base_offset = (y * width + xx) as i32;
        let base_v = data[base_offset as usize];
        let m128_center = [base_v as u8; 16];
        let m128_center =
            _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(&m128_center[0]));
        trace!("m128_center  {}", pi(&m128_center));

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
        let lookup_base = std::mem::transmute::<_, *const i32>(&data[base_offset as usize]);
        let obtained = _mm256_i32gather_epi32(lookup_base, indices, SCALE);

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
        let mut retrievable = [0u8; 16];
        retrievable[0..4].copy_from_slice(&lower.to_le_bytes());
        retrievable[4..8].copy_from_slice(&higher.to_le_bytes());

        // And, in a second gather, we can get us the remaining 8 indices.
        let indices = _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(
            &circle_offset[SOUTH],
        ));
        let lookup_base = std::mem::transmute::<_, *const i32>(&data[base_offset as usize]);
        let obtained = _mm256_i32gather_epi32(lookup_base, indices, SCALE);
        let left_u32_per_lane = _mm256_shuffle_epi8(obtained, mask);
        let lower = _mm256_extract_epi32(left_u32_per_lane, 0);
        let higher = _mm256_extract_epi32(left_u32_per_lane, 4);

        // Which we can also read into retrievable. That now holds our circle pixels.
        retrievable[8..12].copy_from_slice(&lower.to_le_bytes());
        retrievable[12..16].copy_from_slice(&higher.to_le_bytes());

        // Retrievable is correct, confirmed that with some prints.
        trace!("Values dec  {retrievable:?}");

        // Definition of the paper is, let a cirle point be p and center of the circle c.
        // darker: p <= c - t
        // similar: c - t < p < c + t
        // brigher: c + t <= p

        // Load the circle's points.
        let p = _mm_loadu_si128(std::mem::transmute::<_, *const __m128i>(&retrievable[0]));
        trace!("Values hex  {}", pi(&p));

        // Now, we can calculate the lower and upper bounds.
        let upper_bound = _mm_adds_epu8(m128_center, m128_threshold);
        let lower_bound = _mm_subs_epu8(m128_center, m128_threshold);
        trace!("upper_bound {}", pi(&upper_bound));
        trace!("lower_bound {}", pi(&lower_bound));

        let is_above = _mm_cmpgt_epu8(p, upper_bound);
        let is_below = _mm_cmpgt_epu8(lower_bound, p);
        trace!("is_above    {}", pi(&is_above));
        trace!("is_below    {}", pi(&is_below));

        // void _mm256_storeu_si256 (__m256i * mem_addr, __m256i a)
        /*
        let mut above_u8 = [0u8; 32];
        _mm_storeu_si128(
            std::mem::transmute::<_, *mut __m128i>(&above_u8[0]),
            is_above,
        );
        _mm_storeu_si128(
            std::mem::transmute::<_, *mut __m128i>(&above_u8[16]),
            is_above,
        );
        let mut below_u8 = [0u8; 32];
        _mm_storeu_si128(
            std::mem::transmute::<_, *mut __m128i>(&below_u8[0]),
            is_below,
        );
        _mm_storeu_si128(
            std::mem::transmute::<_, *mut __m128i>(&below_u8[16]),
            is_below,
        );
        trace!("above_u8    {above_u8:?}");
        trace!("below_u8    {below_u8:?}");
        */
        const COUNT: usize = 16;

        let below_bits = _mm_movemask_epi8(is_below);
        let above_bits = _mm_movemask_epi8(is_above);
        trace!("below_bits    {below_bits:?}");
        trace!("above_bits    {above_bits:?}");

        let below_count = below_bits.count_ones();
        let above_count = above_bits.count_ones();
        trace!("below_count    {below_count:?}");
        trace!("above_count    {above_count:?}");

        // let below_left = _mm_extract_epi64(is_below, 0);
        // let below_right = _mm_extract_epi64(is_below, 1);
        // let below_bits =

        // There's probably a way more efficient way of doing this rotation.

        // Next, we need to figure out if there is a consecutive sequence of above or below
        // in the ring of 16.

        // Working with that ring however, is tricky with the wraparound.
        // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
        //                                                  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
        // We can solve that problem by concatenating the vector again at the end, and iterating
        // over the section that is (16 + consecutive)
        let mut found_consecutive = 0;

        let used_bits = if below_count > above_count {
            below_bits as u32 | ((below_bits as u32) << 16)
        } else {
            above_bits as u32 | ((above_bits as u32) << 16)
        };
        for k in 0..(COUNT + consecutive as usize) {
            if (used_bits & (1 << k)) != 0 {
                found_consecutive += 1;
                if found_consecutive >= consecutive {
                    return Some(FastPoint { x: xx, y });
                }
            } else {
                found_consecutive = 0;
            }
        }

        None
    }

    pub fn detect(image: &image::GrayImage, t: u8, consecutive: u8) -> Vec<FastPoint> {
        let height = image.height();
        let width = image.width();
        let t = t as i16;
        // exists range n where all entries different than p - t.
        let mut r = vec![];

        let data = image.as_raw();

        // calculate the circle offsets for the data once.
        let circle_offset = calculate_offsets(width);

        const STEP_SIZE: usize = 32;
        unsafe {
            let m128_threshold = [t as u8; STEP_SIZE];
            let m128_threshold =
                _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(&m128_threshold[0]));

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

                // Cardinal directions:
                //  n >= 12: 3/4
                //  n >= 9 : 2/4

                // If that is the case, and only then should we do real work.

                let x_chunks = (width - 3 - 3) / STEP_SIZE as u32;
                for x_step in 0..x_chunks {
                    let x = 3 + x_step * STEP_SIZE as u32;

                    trace!("\n\n");
                    // Ok, here we go.
                    // __m128i = 16 bytes;
                    // __m256i = 32 bytes;
                    let base_offset = (y * width + x) as i32;

                    // Obtain 32 centers.
                    let c = _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(
                        &data[base_offset as usize],
                    ));
                    trace!("c    : {}", pl(&c));
                    trace!("t    : {}", pl(&m128_threshold));

                    // Obtain 16 of the cardinal directions.
                    let north = _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(
                        &data[(base_offset + circle_offset[NORTH]) as usize],
                    ));
                    let east = _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(
                        &data[(base_offset + circle_offset[EAST]) as usize],
                    ));
                    let south = _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(
                        &data[(base_offset + circle_offset[SOUTH]) as usize],
                    ));
                    let west = _mm256_loadu_si256(std::mem::transmute::<_, *const __m256i>(
                        &data[(base_offset + circle_offset[WEST]) as usize],
                    ));
                    trace!("");
                    trace!("north: {}", pl(&north));
                    trace!("east : {}", pl(&east));
                    trace!("south: {}", pl(&south));
                    trace!("west:  {}", pl(&west));

                    // Ok, great, we have the data on hand, now for each byte, we can
                    // Now, we can calculate the lower and upper bounds.
                    let upper_bound = _mm256_adds_epu8(c, m128_threshold);
                    let lower_bound = _mm256_subs_epu8(c, m128_threshold);
                    trace!("");
                    trace!("upbnd: {}", pl(&upper_bound));
                    trace!("lrbnd: {}", pl(&lower_bound));

                    // Now, we just need to determine if 3 out of 4 of the cardinal directions are above or below.
                    // These masks return 0xFF if true, x00 if false.
                    // _mm_cmpgt_epi8: dst[i+7:i] := ( a[i+7:i] > b[i+7:i] ) ? 0xFF : 0

                    let north_above = _mm256_cmpgt_epu8(north, upper_bound);
                    let east_above = _mm256_cmpgt_epu8(east, upper_bound);
                    let south_above = _mm256_cmpgt_epu8(south, upper_bound);
                    let west_above = _mm256_cmpgt_epu8(west, upper_bound);
                    trace!("");
                    trace!("nt_ab: {}", pl(&north_above));
                    trace!("ea_ab: {}", pl(&east_above));
                    trace!("st_ab: {}", pl(&south_above));
                    trace!("we_ab: {}", pl(&west_above));

                    let north_below = _mm256_cmpgt_epu8(lower_bound, north);
                    let east_below = _mm256_cmpgt_epu8(lower_bound, east);
                    let south_below = _mm256_cmpgt_epu8(lower_bound, south);
                    let west_below = _mm256_cmpgt_epu8(lower_bound, west);
                    trace!("");
                    trace!("nt_bl: {}", pl(&north_below));
                    trace!("ea_bl: {}", pl(&east_below));
                    trace!("st_bl: {}", pl(&south_below));
                    trace!("we_bl: {}", pl(&west_below));

                    trace!("");

                    let check_mask = if consecutive < 12 && consecutive >= 9  {
                        // Now, we need a way to determine 2 out of 4.
                        //               && south && west
                        // north &&               && west
                        // north && east &&       &&
                        //       && east && south &&

                        // That has only four options, why not just write it out?
                        // That has only four options, why not just write it out?
                        let above_0 = _mm256_and_si256(south_above, west_above);
                        let above_1 = _mm256_and_si256(north_above, west_above);
                        let above_2 = _mm256_and_si256(north_above, east_above);
                        let above_3 = _mm256_and_si256(east_above, south_above);
                        let above_2_found = _mm256_or_si256(
                            _mm256_or_si256(above_0, above_1),
                            _mm256_or_si256(above_2, above_3),
                        );
                        trace!("2 gtf: {}", pl(&above_2_found));

                        // And the same for below.
                        let below_0 = _mm256_and_si256(south_below, west_below);
                        let below_1 = _mm256_and_si256(north_below, west_below);
                        let below_2 = _mm256_and_si256(north_below, east_below);
                        let below_3 = _mm256_and_si256(east_below, south_below);
                        let below_2_found = _mm256_or_si256(
                            _mm256_or_si256(below_0, below_1),
                            _mm256_or_si256(below_2, below_3),
                        );
                        trace!("2 ltf: {}", pl(&below_2_found));

                        let found_3 = _mm256_or_si256(above_2_found, below_2_found);

                        let mut mask = [0u8; STEP_SIZE];
                        _mm256_storeu_si256(
                            std::mem::transmute::<_, *mut __m256i>(&mut mask[0]),
                            found_3,
                        );
                        mask
                    } else if consecutive >= 12 {
                        // Now, we need a way to determine 3 out of 4.
                        //          east && south && west
                        // north &&         south && west
                        // north && east &&       && west
                        // north && east && south &&

                        // That has only four options, why not just write it out?
                        let above_0 =
                            _mm256_and_si256(_mm256_and_si256(east_above, south_above), west_above);
                        let above_1 =
                            _mm256_and_si256(_mm256_and_si256(north_above, south_above), west_above);
                        let above_2 =
                            _mm256_and_si256(_mm256_and_si256(north_above, east_above), west_above);
                        let above_3 =
                            _mm256_and_si256(_mm256_and_si256(north_above, east_above), south_above);
                        let above_3_found = _mm256_or_si256(
                            _mm256_or_si256(above_0, above_1),
                            _mm256_or_si256(above_2, above_3),
                        );
                        trace!("3 gtf: {}", pl(&above_3_found));

                        // And the same for below.
                        let below_0 =
                            _mm256_and_si256(_mm256_and_si256(east_below, south_below), west_below);
                        let below_1 =
                            _mm256_and_si256(_mm256_and_si256(north_below, south_below), west_below);
                        let below_2 =
                            _mm256_and_si256(_mm256_and_si256(north_below, east_below), west_below);
                        let below_3 =
                            _mm256_and_si256(_mm256_and_si256(north_below, east_below), south_below);
                        let below_3_found = _mm256_or_si256(
                            _mm256_or_si256(below_0, below_1),
                            _mm256_or_si256(below_2, below_3),
                        );
                        trace!("3 ltf: {}", pl(&below_3_found));

                        let found_3 = _mm256_or_si256(above_3_found, below_3_found);

                        let mut mask = [0u8; STEP_SIZE];
                        _mm256_storeu_si256(
                            std::mem::transmute::<_, *mut __m256i>(&mut mask[0]),
                            found_3,
                        );
                        mask
                    } else {
                        [0xFFu8; STEP_SIZE]
                    };

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
                // for i in (input.len() / c) * c..input.len()
                for x_step in
                    ((width - 3 - 3) / STEP_SIZE as u32) * (STEP_SIZE as u32)..(width - 3 - 3)
                {
                    // for x in (width - 16 - 3)..(width -3){
                    let x = x_step + 3;
                    if let Some(keypoint) = determine_keypoint(
                        data,
                        &circle_offset,
                        width,
                        (x, y),
                        t as u8,
                        consecutive,
                    ) {
                        r.push(keypoint);
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

#[cfg(test)]
mod test {
    use super::*;
    fn create_sample_image(center: u8, circle: &[u8]) -> image::GrayImage {
        let w = 128;
        let h = 128;
        let mut z = image::GrayImage::new(w, h);
        z.put_pixel(w / 2, h / 2, Luma::<u8>([center]));
        assert_eq!(circle.len(), 16);
        for (i, v) in circle.iter().enumerate() {
            let offset = fast_detector16::circle()[i];
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

        let height = img.height();
        let width = img.width();
        let circle_offset = fast_detector16::calculate_offsets(width);
        let z = unsafe {
            fast_detector16::determine_keypoint(
                &img.as_raw(),
                &circle_offset,
                width,
                (img.width() / 2, img.height() / 2),
                threshold,
                count_minimum,
            )
        };

        let detected = fast_detector16::detect(&img, 16, 9);
        assert_eq!(
            detected.contains(&FastPoint {
                x: img.width() / 2,
                y: img.height() / 2
            }),
            true
        );
    }
}
