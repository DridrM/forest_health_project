from typing import List, Optional, Union

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Polygon

from fhealth.params import DEFAULT_RGB_BLENDING_RATIO


class ImageDataHandler(BaseModel):
    """
    Pydantic-based data handler to manage RGB images, create masks from polygons,
    blend images.

    Fields:
    ----------
        rgb_image (PIL.Image.Image): The RGB image used as the base for blending and mask creation.
        polygons (List[shapely.geometry.Polygon]): A list of Shapely polygons used to create the mask.
        blending_alpha (float): The alpha value (0 to 1) controlling the blending level between
                                the mask and RGB image (0 = no mask, 1 = full mask).
        mask_image (Optional[PIL.Image.Image]): The black-and-white mask image created from polygons,
                                                initialized to None until generated.
        blended_image (Optional[PIL.Image.Image]): The blended image (RGB + mask), initialized to None
                                                   until the blending is performed.
    """

    rgb_image: Image.Image
    mask_polygons: List[Polygon]
    blending_alpha: Union[float, None] = Field(
        DEFAULT_RGB_BLENDING_RATIO,
        ge=0.0,
        le=1.0,
        description="Blending level between 0 (no mask) and 1 (full mask)",
    )
    mask_image: Optional[Image.Image] = Field(
        None, init=False, description="The black-and-white image representing the mask"
    )
    blended_image: Optional[Image.Image] = Field(
        None, init=False, description="The blended RGB image with the mask"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def blend_mask_with_rgb(self) -> None:
        """
        Blends the current mask image with the RGB image using the specified alpha level.

        If the mask image hasn't been created or the mask and RGB image sizes don't match,
        an error is raised.

        This method updates the `blended_image` attribute with the resulting blended image.

        Raises:
            ValueError: If the mask image is not provided or the mask and RGB image sizes do not match.
        """
        if not self.mask_image:
            raise ValueError("Mask image not provided.")

        # Ensure both images are the same size
        if self.mask_image.size != self.rgb_image.size:
            raise ValueError("RGB image and mask image sizes do not match.")

        # Blend the images using the blending alpha
        blended_image = Image.blend(
            self.rgb_image.convert("RGBA"),
            self.mask_image.convert("RGBA"),
            self.blending_alpha,
        )

        # Convert the RGBA blended image to RGB by discarding the alpha channel
        self.blended_image = blended_image.convert("RGB")

    def create_mask_from_polygons(self) -> None:
        """
        Creates a black-and-white mask image based on the list of Shapely polygons.
        Each polygon is drawn as a white area on a black background.

        This method updates the `mask_image` attribute with the generated mask image.

        Raises:
            ValueError: If no polygons are provided to create the mask.
        """
        if not self.mask_polygons:
            raise ValueError("No polygons provided to create the mask.")

        # Create a blank black-and-white image with the same size as the RGB image
        mask_image = Image.new(
            "L", self.rgb_image.size, 0
        )  # "L" mode is for greyscale (0 = black, 255 = white)
        draw = ImageDraw.Draw(mask_image)

        # Loop through each polygon and draw it on the mask image
        for polygon in self.mask_polygons:
            # Convert polygon to a list of tuples that ImageDraw can use
            xy = list(polygon.exterior.coords)
            draw.polygon(xy, outline=255, fill=255)

        self.mask_image = mask_image

    def downgrade_resolution(self, target_size: int) -> None:
        """
        Downgrades the resolution of the mask image and blended image to the specified target size,
        keeping them in a square format.

        Args:
            target_size (int): The target size for both the width and height of the images (in pixels).

        Raises:
            ValueError: If the mask image or blended image is not available to resize.
        """
        if self.mask_image is None:
            raise ValueError("Mask image not available for resolution downgrade.")

        if self.blended_image is None:
            raise ValueError("Blended image not available for resolution downgrade.")

        # Resize both the mask image and the blended image to the target square size
        self.mask_image = self.mask_image.resize((target_size, target_size))
        self.blended_image = self.blended_image.resize((target_size, target_size))
