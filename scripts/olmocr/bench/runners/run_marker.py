from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

_marker_converter = None


def run_marker(pdf_path: str, page_num: int = 1) -> str:
    global _marker_converter

    if _marker_converter is None:
        _marker_converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )

    rendered = _marker_converter(pdf_path)
    # Create the markdown filename by replacing the .pdf extension with .md
    text, _, images = text_from_rendered(rendered)

    return text
