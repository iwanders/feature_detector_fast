use image::{Rgb, GenericImageView, Luma};

pub struct Rgb8ToLuma16View<'a> {
    img: &'a dyn GenericImageView<Pixel=Rgb<u8>>,
}

impl<'a> Rgb8ToLuma16View<'a> {
    pub fn new(img: &'a dyn GenericImageView<Pixel=Rgb<u8>>) -> Self {
        Rgb8ToLuma16View{img}
    }

    pub fn to_grey(&self) -> image::ImageBuffer<Luma<u8>, Vec<u8>> {
        use image::GenericImage;
        let mut imageBuffer = image::ImageBuffer::<Luma<u8>, Vec<u8>>::new(self.img.dimensions().0, self.img.dimensions().1);

        for (x, y, _pixel) in self.pixels() {
            imageBuffer.put_pixel(x, y, Luma([(self.get_pixel(x, y)[0] / 3) as u8]));
        }
        imageBuffer
    }
}

impl<'a>  image::GenericImageView  for Rgb8ToLuma16View<'a> {
    type Pixel = Luma<u16>;

    fn dimensions(&self) -> (u32, u32) {
        self.img.dimensions()
    }
    fn bounds(&self) -> (u32, u32, u32, u32){
        self.img.bounds()
    }
    fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel{
        let rgb = self.img.get_pixel(x, y);
        Luma([rgb[0] as u16 + rgb[1] as u16 + rgb[2] as u16])
    }
}

pub const WHITE: Rgb<u8> = Rgb([255u8, 255u8, 255u8]);
pub const RED: Rgb<u8> = Rgb([255u8, 0u8, 0u8]);
pub const GREEN: Rgb<u8> = Rgb([0u8, 255u8, 0u8]);
pub const BLUE: Rgb<u8> = Rgb([0u8, 0u8, 255u8]);

pub fn draw_plus<I>(image: &mut I, (x, y): (u32, u32), color: I::Pixel)
where
    I: image::GenericImage,
    I::Pixel: 'static,
{
    draw_plus_sized(image, (x, y), color, 3);
}

pub fn draw_plus_sized<I>(image: &mut I, (x, y): (u32, u32), color: I::Pixel, size: u32)
where
    I: image::GenericImage,
    I::Pixel: 'static,
{
    let (w, h)  = image.dimensions();
    let s = size as i32;
    for d in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
        for l in 0..s {
            let dx = d.0 * l;
            let dy = d.1 * l;
            let px = (x as i32) + dx;
            let py = (y as i32) + dy;
            if (px <= 0 || px >= w as i32) || (py <= 0 || py >= h as i32) {
                continue;
            }
            image.put_pixel(px as u32, py as u32, color);
        }
    }
}
