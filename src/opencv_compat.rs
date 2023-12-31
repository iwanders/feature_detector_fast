/*!

This implementation aims to exactly match the OpenCV implementation of FAST.

The original author's source code can be found here:
    <https://web.archive.org/web/20070708064606/http://mi.eng.cam.ac.uk/~er258/work/fast.html>

It looks like the decision tree as described in rosten2006.pdf "LNCS 3951 - Machine Learning for High-Speed Corner Detection" by Rosten and Drummond, not core algorithm.
This reference implementation's nonmax suppression can have three keypoints adjecent on a single row, which violates the 9x9 non max suppression guarantees?


The opencv (version 3.2) implementation does not match the original author's decision tree, because it is a raw implementation of the algorithm instead of a trained decision tree.

OpenCV's nonmax score function appears to be the threshold for which the point would still be a keypoint.
    Which the paper states a lot of pixels will share the value, they propose an alternative.
    Enabling VERIFY_CORNERS in the OpenCV code makes the asserts fail.

OpenCV enshrines 9 consecutive pixels out of the 16 all throughout the code base, eliminating the possibility of using n >= 12 for which there is a 3 / 4 cardinal direction check.


In this file, I implemented (very naively) logic that is identical to opencv:
    - The detect() without non maximal supression here matches opencv for a count of 9.
    - The non_max_supression_opencv method matches opencv's nonmax suppression for a count of 9.

*/
use crate::{Config, Point};
use image::{GenericImageView, Luma};

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

/// Retrieve a point on the circle.
const fn point(index: u8) -> (i32, i32) {
    circle()[index as usize % circle().len()]
}

/// Create a blue circle of the points for debugging.
pub fn make_circle_image() -> image::RgbImage {
    const BLUE: image::Rgb<u8> = image::Rgb([0u8, 0u8, 255u8]);
    let mut image = image::RgbImage::new(32, 32);
    for (dx, dy) in circle().iter() {
        image.put_pixel((16i32 + dx) as u32, (16i32 + dy) as u32, BLUE)
    }
    image
}

/// Perform the detection on a grayscale image.
pub fn detect(
    image: &dyn GenericImageView<Pixel = Luma<u8>>,
    t: u8,
    consecutive: u8,
) -> Vec<Point> {
    let t = t as i16;

    let (width, height) = image.dimensions();

    let mut r = vec![];

    for y in 3..(height - 3) {
        for x in 3..(width - 3) {
            // exists range n where all entries different than p - t.

            let base_v = image.get_pixel(x, y)[0];
            trace!("{y}, {x}");
            trace!("   {base_v} ");

            let delta_f = |index: usize| {
                let offset = point(index as u8);
                let t_x = (x as i32 + offset.0) as u32;
                let t_y = (y as i32 + offset.1) as u32;
                let pixel_v = image.get_pixel(t_x, t_y)[0];

                let delta = base_v as i16 - pixel_v as i16;
                delta
            };
            let p_f = |index: usize| {
                let offset = point(index as u8);
                let t_x = (x as i32 + offset.0) as u32;
                let t_y = (y as i32 + offset.1) as u32;
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

            // There's a way more efficient way of doing this rotation, kept like this because it
            // is a direct translation of finding the number of consecutive values that exceed
            // the threshold on a ringbuffer.
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
                    r.push(Point { x, y });
                    break;
                }
            }
        }
    }
    r
}

/// Calculate the non maximal suppression score that OpenCV would calculate.
pub fn non_max_suppression_opencv_score(
    image: &image::GrayImage,
    (x, y): (u32, u32),
    count: u8,
) -> u16 {
    // Definition of the paper is, let a cirle point be p and center of the circle c.
    //     darker: p <= c - t
    //     similar: c - t < p < c + t
    //     brigher: c + t <= p
    //
    let base_v = image.get_pixel(x, y)[0] as i16;

    // Opencv has hardcoded 9/16, so their wrap-around ringbuffer is 16 + 9 = 25 long.
    let mut difference = [0i16; 32];
    let offsets = circle();
    for i in 0..difference.len() {
        let pos = circle()[i % offsets.len()];
        let circle_p =
            image.get_pixel((x as i32 + pos.0) as u32, (y as i32 + pos.1) as u32)[0] as i16;
        difference[i] = base_v as i16 - circle_p;
    }

    // OpenCV calculates the highest / lowest extremum across any consecutive block of 9 pixels.
    let mut extreme_highest = std::i16::MIN;
    for k in 0..16 {
        let min_value_of_9 = *difference[k..(k + count as usize)].iter().min().unwrap();
        extreme_highest = extreme_highest.max(min_value_of_9);
    }

    let mut extreme_lowest = std::i16::MAX;
    for k in 0..16 {
        let max_value_of_9 = *difference[k..(k + count as usize)].iter().max().unwrap();
        extreme_lowest = extreme_lowest.min(max_value_of_9);
    }

    // Take the absolute minimum of both to determine the max 't' for which this is a point.
    extreme_highest.abs().min(extreme_lowest.abs()) as u16
}

/// Perform the non max suppression on an a vector of keypoints.
pub fn non_max_supression(
    image: &image::GrayImage,
    keypoints: Vec<Point>,
    config: &crate::Config,
) -> Vec<Point> {
    if config.non_maximal_supression == crate::NonMaximalSuppression::Off {
        return keypoints;
    }

    let score_function: Box<dyn Fn((u32, u32)) -> u16> = match config.non_maximal_supression {
        crate::NonMaximalSuppression::MaxThreshold => {
            Box::new(|p: (u32, u32)| non_max_suppression_opencv_score(image, p, config.count))
        }
        crate::NonMaximalSuppression::SumAbsolute => {
            Box::new(|p: (u32, u32)| non_max_suppression_max_abs(image, p, config.threshold))
        }
        crate::NonMaximalSuppression::Off => {
            unreachable!()
        }
    };

    // Very inefficient.
    let mut res = vec![];

    'kpiter: for kp in keypoints.iter() {
        let current_score = score_function((kp.x, kp.y));
        if kp.y == 3 || kp.y == image.height() - 4 {
            continue;
        }
        for dx in [-1i32, 0, 1] {
            for dy in [-1i32, 0, 1] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                // check if this keypoint exists.
                let zx = (kp.x as i32 + dx) as u32;
                let zy = (kp.y as i32 + dy) as u32;
                if !keypoints.contains(&Point { x: zx, y: zy }) {
                    continue;
                }

                let other_score = score_function((zx, zy));
                if current_score <= other_score {
                    continue 'kpiter;
                }
            }
        }
        res.push(*kp);
    }
    res
}

/// Interface method for the non max calculation.
fn non_max_suppression_max_abs(image: &image::GrayImage, (x, y): (u32, u32), t: u8) -> u16 {
    let base_v = image.get_pixel(x, y)[0];

    let mut values = [0u8; 16];
    for i in 0..values.len() {
        let pos = circle()[i];
        let circle_p = image.get_pixel((x as i32 + pos.0) as u32, (y as i32 + pos.1) as u32)[0];
        values[i] = circle_p;
    }
    score_non_max_supression_max_abs_sum(base_v, &values, t)
}

/// Non max equation number 3 from the paper.
pub fn score_non_max_supression_max_abs_sum(base_v: u8, circle: &[u8], t: u8) -> u16 {
    // println!("              base: {base_v:02x}, t: {t:02x}, circle: {circle:02x?}");
    let mut sum_dark: u16 = 0;
    let mut sum_light: u16 = 0;
    assert_eq!(circle.len(), 16);
    let mut _values_dark = [0u8; 16];
    let mut _values_light = [0u8; 16];
    for i in 0..circle.len() {
        let d = base_v as i16 - circle[i] as i16;
        if d > 0 && d.abs() > (t as i16) {
            let value = (base_v - circle[i]) - t;
            _values_light[i] = value;
            sum_light += value as u16;
        }
        if d < 0 && d.abs() > (t as i16) {
            let value = (circle[i] - base_v) - t;
            _values_dark[i] = value;
            sum_dark += value as u16;
        }
    }
    sum_dark.max(sum_light)
}

/// Run the feature detector given the provided image and configuration.
pub fn detector(img: &image::GrayImage, config: &Config) -> Vec<Point> {
    let r = detect(img, config.threshold, config.count);

    non_max_supression(img, r, config)
}

#[cfg(test)]
mod test {
    #[test]
    fn test_consecutive() {
        fn test_consecutive(z: &[u8], consecutive: usize) -> bool {
            for s in 0..z.len() {
                if z.iter()
                    .map(|v| *v != 0)
                    .cycle()
                    .skip(s)
                    .take_while(|t| *t)
                    .count()
                    >= consecutive as usize
                {
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

        assert_eq!(
            test_consecutive(&[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 3),
            false
        );

        assert_eq!(
            test_consecutive(&[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], 4),
            true
        );
    }
}
