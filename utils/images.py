import base64
import io
from PIL import Image


def image_to_base64(image: Image.Image) -> tuple[str, str] | bool:
    buffer = io.BytesIO()
    format = getattr(image, "format")

    if format is None:
        return False

    if format not in ["JPEG", "PNG", "WEBP", "GIF"]:
        return False

    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return img_str, format.lower()
