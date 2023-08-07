/*!

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
use crate::{FastConfig, FastPoint};
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

pub fn detect(
    image: &dyn GenericImageView<Pixel = Luma<u8>>,
    t: u8,
    consecutive: u8,
) -> Vec<FastPoint> {
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
                    r.push(FastPoint { x, y });
                    break;
                }
            }
        }
    }
    r
}

pub fn non_max_suppression_opencv_score(image: &image::GrayImage, (x, y): (u32, u32)) -> i16 {
    // Definition of the paper is, let a cirle point be p and center of the circle c.
    //     darker: p <= c - t
    //     similar: c - t < p < c + t
    //     brigher: c + t <= p
    //
    let base_v = image.get_pixel(x, y)[0] as i16;

    // Opencv has hardcoded 9/16, so their wrap-around ringbuffer is 16 + 9 = 25 long.
    let mut difference = [0i16; 25];
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
        let min_value_of_9 = *difference[k..(k + 9)].iter().min().unwrap();
        extreme_highest = extreme_highest.max(min_value_of_9);
    }

    let mut extreme_lowest = std::i16::MAX;
    for k in 0..16 {
        let max_value_of_9 = *difference[k..(k + 9)].iter().max().unwrap();
        extreme_lowest = extreme_lowest.min(max_value_of_9);
    }

    // Take the absolute minimum of both to determine the max 't' for which this is a point.
    extreme_highest.abs().min(extreme_lowest.abs())
}

/// This is identical to opencv, very inefficient though.
pub fn non_max_supression_opencv(
    image: &image::GrayImage,
    keypoints: &[FastPoint],
) -> Vec<FastPoint> {
    // Very inefficient.
    let mut res = vec![];

    'kpiter: for kp in keypoints.iter() {
        let current_score = non_max_suppression_opencv_score(image, (kp.x, kp.y));
        if kp.x == 3 || kp.x == image.width() - 4 {
            continue 'kpiter;
        }
        if kp.y == 3 || kp.y == image.height() - 4 {
            continue 'kpiter;
        }
        for dx in [-1i32, 0, 1] {
            for dy in [-1i32, 0, 1] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                // check if this keypoint exists.
                let zx = (kp.x as i32 + dx) as u32;
                let zy = (kp.y as i32 + dy) as u32;
                if !keypoints.contains(&FastPoint { x: zx, y: zy }) {
                    continue;
                }

                let other_score = non_max_suppression_opencv_score(image, (zx, zy));
                if current_score <= other_score {
                    continue 'kpiter;
                }
            }
        }
        res.push(*kp);
    }
    res
}

pub fn detector(img: &image::GrayImage, config: &FastConfig) -> Vec<FastPoint> {
    let r = detect(img, config.threshold, config.count);

    if config.non_maximal_supression {
        return non_max_supression_opencv(img, &r);
    }
    r
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
