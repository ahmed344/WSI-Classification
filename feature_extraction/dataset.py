from torch.utils.data import Dataset

class WSI_tiles(Dataset):
    def __init__(self, slide, tiles, size, transform=None):
        """Initialize the WSI tiles dataset

        Args:
            slide (OpenSlide): Whole slide image (WSI)
            tiles (DataFrame): WSI tiles centers with x and y columns
            transform (torchvision.transforms.transforms.Compose, optional): Transform function. Defaults to None.
        """
        self.slide = slide 
        self.tiles = tiles
        self.x = self.tiles.x.round().astype(int)
        self.y = self.tiles.y.round().astype(int)
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
            torch.Tensor: Image tile
        """
        # Extract a tile from the slide
        tile = self.slide.read_region(location=(self.x[idx], self.y[idx]),
                                      level=0,
                                      size=self.size)

        # Convert the tile to RGB
        tile = tile.convert("RGB")

        # Transform the tile using the transform function
        tile = self.transform(tile)

        return tile