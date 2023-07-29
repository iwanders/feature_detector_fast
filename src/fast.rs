use image::{GenericImageView, Luma};

#[derive(Copy, Debug, Clone)]
pub struct FastPoint {
    pub x: u32,
    pub y: u32,
}

pub mod fast_detector16 {
    const DO_PRINTS: bool = true;

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
        (x, y): (u32, u32),
        image: &dyn GenericImageView<Pixel = Luma<u8>>,
        t: u16,
        consecutive: u8,
    ) -> Option<FastPoint> {
        // exists range n where all entries different than p - t.

        let base_v = image.get_pixel(x, y)[0];
        trace!("{y}, {x}");
        trace!("   {base_v} ");

        let delta_f = |index: usize| {
            let offset = point(index as u8);
            let t_x = (x as i32 + offset.0) as u32;
            let t_y = (y as i32 + offset.1) as u32;
            // Using unsafe here shaves off ~15%.
            let pixel_v = image.get_pixel(t_x, t_y)[0];

            let delta = base_v as i16 - pixel_v as i16;
            delta
        };
        let p_f = |index: usize| {
            let offset = point(index as u8);
            let t_x = (x as i32 + offset.0) as u32;
            let t_y = (y as i32 + offset.1) as u32;
            // Using unsafe here shaves off ~15%.
            image.get_pixel(t_x, t_y)[0]
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

        if DO_PRINTS {
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
                return Some(FastPoint { x, y });
            }
        }

        None
    }
}

pub struct FastConfig {
    /// Value to be exceeded.
    pub threshold: u8,
    /// Count of consecutive pixels
    pub count: u8,
}

pub fn detector(
    img: &dyn GenericImageView<Pixel = Luma<u8>>,
    config: &FastConfig,
) -> Vec<FastPoint> {
    // let luma_view = crate::util::Rgb8ToLuma16View::new(img);
    let (width, height) = img.dimensions();

    for y in 3..(height - 3) {
        for x in 3..(width - 3) {
            println!("{x}, {y}, {}", img.get_pixel(x, y)[0]);
        }
    }


    let mut r = vec![];

    for y in 3..(height - 3) {
        for x in 3..(width - 3) {
            if let Some(p) =
                fast_detector16::detect((x, y), img, config.threshold as u16 , config.count)
            {
                r.push(p);
            }
        }
    }
    r
}

#[cfg(test)]
mod test{
    #[test]
    fn test_consecutive() {
        fn test_consecutive(z: &[u8], consecutive: usize) -> bool {
            for s in 0..z.len(){
                if z
                .iter()
                .map(|v| *v != 0)
                .cycle()
                .skip(s)
                .take_while(|t| *t)
                .count()
                >= consecutive as usize {
                    return true;
                }
            }
            false
        }
        assert_eq!(test_consecutive(&[0, 0, 0, 1], 3), false);
        assert_eq!(test_consecutive(&[1, 0, 0, 1], 3), false);

        assert_eq!(test_consecutive(&[1, 0, 1, 1], 2), true);


        assert_eq!(test_consecutive(&[0, 1, 1, 1], 3), true);
        assert_eq!(test_consecutive(&[1, 0, 1, 1], 3), true);
        assert_eq!(test_consecutive(&[1, 1, 0, 1], 3), true);
        assert_eq!(test_consecutive(&[1, 1, 1, 0], 3), true);

        assert_eq!(test_consecutive(&[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 3), false);

        assert_eq!(test_consecutive(&[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], 4), true);
    }
}
