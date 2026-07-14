from dataclasses import dataclass

Color = tuple[int, int, int]

@dataclass
class Arguments:

    # texture
    DENSITY_APPLICABLE_TEXTURES = ["texture_D", "texture_G", "texture_I", "texture_J"]
    texture_name: str
    density_applicable: bool

    # texture modifications
    count: int
    scale_x: float
    scale_y: float
    density_x: float
    density_y: float
    rotation: float
    global_scale: float

    # colors
    colors: list[Color]
    color_strengths: list[float]
    white_background: bool
    background_brightness: float

    # square texture
    generate_square: bool
    width: int
    height: int

    seed: int

    def __init__(self,
                 texture_name: str = "texture_J",
                 count: int = 1,
                 scale_x: float = 1.0,
                 scale_y: float = 1.0,
                 density_x: float = 0.0,
                 density_y: float = 0.0,
                 rotation: float = 0.0,
                 global_scale: float = 1.0,
                 colors: list[Color] = [(0, 0, 0)],
                 color_strengths: list[float] = [1.0],
                 white_background: bool = True,
                 background_brightness: float = 1.0,
                 generate_square: bool = False,
                 width: int = 1024,
                 height: int = 1024,
                 seed: int = 42
                 ):

        self.texture_name = texture_name
        self.density_applicable = self.texture_name in self.DENSITY_APPLICABLE_TEXTURES
        self.count = count
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.density_x = density_x
        self.density_y = density_y
        self.rotation = rotation
        self.global_scale = global_scale
        self.colors = colors
        self.color_strengths = color_strengths
        self.white_background = white_background
        self.background_brightness = background_brightness
        self.generate_square = generate_square
        self.width = width
        self.height = height
        self.seed = seed


    def clamp_arguments(self):
        self.scale_x = max(0.001, min(self.scale_x, 2.0))
        self.scale_y = max(0.001, min(self.scale_y, 2.0))
        self.density_x = max(-1.0, min(self.density_x, 1.0))
        self.density_y = max(-1.0, min(self.density_y, 1.0))
        self.global_scale = max(0.001, min(self.global_scale, 5.0))
