import openslide
from torch.utils.data import Dataset

class WSI_tiles(Dataset):
    def __init__(self, slide_path, tiles, size, transform=None):
        """Initialize the WSI tiles dataset

        Args:
            slide_path (str): Path to the whole slide image (WSI)
            tiles (DataFrame): WSI tiles centers with x and y columns
            size (tuple[int, int]): Patch size to extract from the WSI
            transform (torchvision.transforms.transforms.Compose, optional): Transform function. Defaults to None.

        Returns:
            None: This initializer does not return a value.
        """
        self.slide_path = slide_path
        self.slide = None
        self.tiles = tiles
        self.x = self.tiles.x.round().to_numpy().astype(int)
        self.y = self.tiles.y.round().to_numpy().astype(int)
        self.transform = transform  # Transform function
        self.size = size


    def __len__(self):
        """Return the number of pixels

        Returns:
            int: Number of pixels
        """
        return self.tiles.shape[0]


    def __getitem__(self, idx):
        """Return the image tile at the given index

        Args:
            idx (int): Index of the image tile

        Returns:
            torch.Tensor: Transformed image tile tensor.
        """
        if not hasattr(self, "slide") or self.slide is None:
            self.slide = openslide.OpenSlide(self.slide_path)

        # Extract a tile from the slide
        tile = self.slide.read_region(location=(self.x[idx], self.y[idx]),
                                      level=0,
                                      size=self.size)

        # Convert the tile to RGB
        tile = tile.convert("RGB")

        # Transform the tile using the transform function
        tile = self.transform(tile)

        return tile