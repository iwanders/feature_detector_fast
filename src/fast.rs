use image::{GenericImageView, Luma};

#[derive(Copy, Debug, Clone)]
pub struct FastPoint {
    pub x: u32,
    pub y: u32,
}

pub mod fast_detector16 {
    const PRINT: bool = false;

    use super::*;

    pub const NORTH: usize = 0;
    pub const EAST: usize = 4;
    pub const SOUTH: usize = 8;
    pub const WEST: usize = 12;

    /// The circle with 16 pixels.
    pub const fn circle() -> [(i32, i32); 16] {
        [
            (0, 3),
            (1, 3),
            (2, 2),
            (3, 1),
            (3, 0),
            (3, -1),
            (2, -2),
            (1, -3),
            (0, -3),
            (-1, -3),
            (-2, -2),
            (-3, -1),
            (-3, 0),
            (-3, 1),
            (-2, 2),
            (-1, 3),
        ]
    }

    pub const fn point(index: u8) -> (i32, i32) {
        circle()[index as usize % circle().len()]
    }

    pub fn detect12(
        (x, y): (u32, u32),
        image: &dyn GenericImageView<Pixel = Luma<u8>>,
        t: u16,
    ) -> Option<FastPoint> {
        // exists range n where all entries different than p - t.

        let base_v = image.get_pixel(x, y)[0];

        let nt = -1 * (t as i16);
        let pt = t as i16;

        let delta_f = |index: usize| {
            let offset = point(index as u8);
            let t_x = (x as i32 + offset.0) as u32;
            let t_y = (y as i32 + offset.1) as u32;
            // Using unsafe here shaves off ~15%.
            let pixel_v = unsafe { image.unsafe_get_pixel(t_x, t_y)[0] };

            let delta = pixel_v as i16 - base_v as i16;
            delta
        };

        // Hmm... 16 directions is 4 * u32.
        // What if we assign the cardinal directions first with rest zeros, then do a 4 integer
        // compare...

        // Implement the cardinal directions shortcut
        let deltas = [delta_f(NORTH), delta_f(EAST), delta_f(SOUTH), delta_f(WEST)];

        // Expand the rotated stuff.
        let three_neg1 = deltas[0] < nt || deltas[1] < nt || deltas[2] < nt;
        let three_neg2 = deltas[1] < nt || deltas[2] < nt || deltas[3] < nt;
        let three_neg3 = deltas[2] < nt || deltas[3] < nt || deltas[0] < nt;
        let three_neg4 = deltas[3] < nt || deltas[0] < nt || deltas[1] < nt;

        let three_pos1 = deltas[0] > pt || deltas[1] > pt || deltas[2] > pt;
        let three_pos2 = deltas[1] > pt || deltas[2] > pt || deltas[3] > pt;
        let three_pos3 = deltas[2] > pt || deltas[3] > pt || deltas[0] > pt;
        let three_pos4 = deltas[3] > pt || deltas[0] > pt || deltas[1] > pt;

        let negative_known = three_neg1 || three_neg2 || three_neg3 || three_neg4;
        let positive_known = three_pos1 || three_pos2 || three_pos3 || three_pos4;

        if !(negative_known || positive_known) {
            return None;
        }

        const COUNT: usize = circle().len() as usize;
        let mut mask = [false; COUNT];

        if negative_known {
            for i in 0..COUNT {
                let d = delta_f(i);
                mask[i] = d < nt;
            }
        } else {
            // Not negative, thus is must be positive.
            for i in 0..COUNT {
                let d = delta_f(i);
                mask[i] = d > pt;
            }
        }

        // This here is less than ideal, we should be able to do it in one pass
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0

        let consecutive = 12;
        for s in 0..COUNT {
            let n = mask
                .iter()
                .cycle()
                .skip(s)
                .take(COUNT)
                .skip_while(|t| !**t)
                .take_while(|t| **t)
                .count()
                > consecutive as usize;

            if n {
                if PRINT {
                    // println!("Succceed by p: {p}, n: {n} at s {s}");
                }
                return Some(FastPoint { x, y });
            }
        }

        None
    }

    pub fn detect(
        (x, y): (u32, u32),
        image: &dyn GenericImageView<Pixel = Luma<u8>>,
        t: u16,
        consecutive: u8,
    ) -> Option<FastPoint> {
        // exists range n where all entries different than p - t.

        let base_v = image.get_pixel(x, y)[0];

        let delta_f = |index: usize| {
            let offset = point(index as u8);
            let t_x = (x as i32 + offset.0) as u32;
            let t_y = (y as i32 + offset.1) as u32;
            // Using unsafe here shaves off ~15%.
            let pixel_v = unsafe { image.unsafe_get_pixel(t_x, t_y)[0] };

            let delta = pixel_v as i16 - base_v as i16;
            delta
        };

        // Implement the cardinal directions shortcut
        let deltas = [delta_f(NORTH), delta_f(EAST), delta_f(SOUTH), delta_f(WEST)];

        let consecutive_cardinal = consecutive / 4;

        let negative = deltas
            .iter()
            .map(|x| x < &0 && x.abs() >= t as i16)
            .skip_while(|t| !t)
            .take_while(|t| *t)
            .count()
            >= consecutive_cardinal as usize;
        let positive = deltas
            .iter()
            .map(|x| x > &0 && x.abs() >= t as i16)
            .skip_while(|t| !t)
            .take_while(|t| *t)
            .count()
            >= consecutive_cardinal as usize;
        if !(negative || positive) {
            return None;
        }

        const COUNT: usize = circle().len() as usize;
        let mut neg = [false; COUNT];
        let mut pos = [false; COUNT];
        for i in 0..COUNT {
            let d = delta_f(i);
            let a = d.abs();
            neg[i] = d < 0 && a >= t as i16;
            pos[i] = d > 0 && a >= t as i16;
        }

        if PRINT {
            for i in 0..COUNT {
                print!("{} ", delta_f(i));
            }
            println!(" t: {t}");
            println!("neg: {neg:?}");
            println!("pos: {pos:?}");
        }

        for s in 0..COUNT {
            let n = neg
                .iter()
                .cycle()
                .skip(s)
                .take(COUNT)
                .skip_while(|t| !**t)
                .take_while(|t| **t)
                .count()
                > consecutive as usize;
            let p = pos
                .iter()
                .cycle()
                .skip(s)
                .take(COUNT)
                .skip_while(|t| !**t)
                .take_while(|t| **t)
                .count()
                > consecutive as usize;

            if n || p {
                if PRINT {
                    println!("Succceed by p: {p}, n: {n} at s {s}");
                }
                return Some(FastPoint { x, y });
            }
        }

        None
    }
}

pub struct FastConfig {
    /// Value to be exceeded.
    pub thresshold: u8,
    /// Count of consecutive pixels
    pub count: u8,
}

pub fn detector(
    img: &dyn GenericImageView<Pixel = Luma<u8>>,
    config: &FastConfig,
) -> Vec<FastPoint> {
    // let luma_view = crate::util::Rgb8ToLuma16View::new(img);

    let (width, height) = img.dimensions();

    let mut r = vec![];

    for y in 3..(height - 3) {
        for x in 3..(width - 3) {
            if let Some(p) =
                fast_detector16::detect((x, y), img, config.thresshold as u16 , config.count)
            {
                r.push(p);
            }
        }
    }
    r
}

pub fn detector12(
    img: &dyn GenericImageView<Pixel = Luma<u8>>,
    config: &FastConfig,
) -> Vec<FastPoint> {
    // let luma_view = crate::util::Rgb8ToLuma16View::new(img);

    let (width, height) = img.dimensions();

    let mut r = vec![];

    for y in 3..(height - 3) {
        for x in 3..(width - 3) {
            if let Some(p) = fast_detector16::detect12((x, y), img, config.thresshold as u16 ) {
                r.push(p);
            }
        }
    }
    r
}
