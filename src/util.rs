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

        for (x, y, pixel) in self.pixels() {
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
