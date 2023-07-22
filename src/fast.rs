use image::{GenericImageView, Luma, Rgb};

#[derive(Copy, Debug, Clone)]
pub struct FastPoint {
    pub x: u32,
    pub y: u32,
}

pub mod fast_detector16 {
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

    pub const fn point(start: u8, index: u8) -> (i32, i32) {
        circle()[(start + index) as usize % circle().len()]
    }

    pub fn detect(
        (x, y): (u32, u32),
        image: &dyn GenericImageView<Pixel = Luma<u16>>,
        t: u16,
        n: u8,
    ) -> Option<FastPoint> {
        // exists range n where all entries different than p - t.

        let base_v = image.get_pixel(x, y)[0];
        let delta_f = |index: usize| {
            let offset = point(0, index as u8);
            let t_x = (x as i32 + offset.0) as u32;
            let t_y = (y as i32 + offset.1) as u32;
            let pixel_v = image.get_pixel(t_x, t_y)[0];

            let delta = pixel_v as i16 - base_v as i16;
            delta
        };

        // Implement the cardinal directions shortcut

        const COUNT: usize = circle().len() as usize;
        let mut neg = [false; COUNT];
        let mut pos = [false; COUNT];

        for ind in [NORTH, EAST, SOUTH, WEST] {
            let d = delta_f(ind);
            let a = d.abs();
            
            neg[ind] = d < 0 && a >= t as i16;
            pos[ind] = d > 0 && a >= t as i16;
        }
        
        // let deltas = [delta_f(NORTH), delta_f(EAST), delta_f(SOUTH), delta_f(WEST)];

        let min_count = n / 4;

        let spot_neg = [&neg[NORTH], &neg[EAST], &neg[SOUTH], &neg[WEST]]; 
        let spot_pos = [&pos[NORTH], &pos[EAST], &pos[SOUTH], &pos[WEST]]; 

        let negative = spot_neg.iter()
            .skip_while(|t| !**t)
            .take_while(|t| ***t)
            .count()
            >= min_count as usize;
        let positive = spot_pos.iter()
            .skip_while(|t| !**t)
            .take_while(|t| ***t)
            .count()
            >= min_count as usize;
        if !(negative || positive) {
            return None;
        }

        for i in 0..COUNT {
            if [NORTH, EAST, SOUTH, WEST].contains(&i) {
                continue;
            }
            let d = delta_f(i);
            let a = d.abs();
            neg[i] = d < 0 && a >= t as i16;
            pos[i] = d > 0 && a >= t as i16;
        }

        for s in 0..COUNT {
            let t = neg
                .iter()
                .cycle()
                .skip(s)
                .take(COUNT)
                .skip_while(|t| !**t)
                .take_while(|t| **t)
                .count()
                > n as usize;
            if t {
                return Some(FastPoint { x, y });
            }
            let t = pos
                .iter()
                .cycle()
                .skip(s)
                .take(COUNT)
                .skip_while(|t| !**t)
                .take_while(|t| **t)
                .count()
                > n as usize;
            if t {
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
    img: &dyn GenericImageView<Pixel = Rgb<u8>>,
    config: &FastConfig,
) -> Vec<FastPoint> {
    let luma_view = crate::util::Rgb8ToLuma16View::new(img);

    let (width, height) = img.dimensions();

    let mut r = vec![];

    for y in 3..(height - 3) {
        for x in 3..(width - 3) {
            if let Some(p) = fast_detector16::detect(
                (x, y),
                &luma_view,
                config.thresshold as u16 * 3,
                config.count,
            ) {
                r.push(p);
            }
        }
    }
    r
}
