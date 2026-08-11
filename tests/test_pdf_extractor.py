from pathlib import Path

from app.ingestion.extractors.pdf_extractor import PDFExtractor


def main():
    extractor = PDFExtractor()

    raw_document = extractor.extract(
        Path("storage/uploads/PDF_BenchmarkTester_63Pages.pdf")
    )

    print("=" * 50)
    print(f"File Name   : {raw_document.file_name}")
    print(f"Total Pages : {raw_document.total_pages}")
    print(f"Pages Found : {len(raw_document.pages)}")
    print("=" * 50)

    print(raw_document.pages[0])


if __name__ == "__main__":
    main()