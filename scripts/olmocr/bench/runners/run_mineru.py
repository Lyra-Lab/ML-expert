import os
import tempfile

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze


def run_mineru(pdf_path: str, page_num: int = 1) -> str:
    output_folder = tempfile.TemporaryDirectory()
    image_output_folder = tempfile.TemporaryDirectory()

    # Initialize writers (same for all PDFs)
    image_writer = FileBasedDataWriter(image_output_folder.name)
    md_writer = FileBasedDataWriter(output_folder.name)

    # Read the PDF file bytes
    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(pdf_path)

    # Create dataset instance
    ds = PymuDocDataset(pdf_bytes)

    # Inference: decide whether to run OCR mode based on dataset classification
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    # Generate markdown content; the image directory is the basename of the images output folder
    image_dir_basename = os.path.basename(image_output_folder.name)
    # md_content = pipe_result.get_markdown(image_dir_basename)

    # Dump markdown file
    with tempfile.NamedTemporaryFile("w+", suffix="md") as tf:
        pipe_result.dump_md(md_writer, tf.name, image_dir_basename)
        tf.flush()

        tf.seek(0)
        md_data = tf.read()

    return md_data
