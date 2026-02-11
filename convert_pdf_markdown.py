
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from pathlib import Path

def analyze_pdf_to_markdown(pdf_path: str, endpoint: str, key: str) -> str:
    """
    Analyze a PDF document and convert it to Markdown format.
    """
    # Initialize the client
    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    # Read the PDF file
    with open(pdf_path, "rb") as f:
        # Analyze the document using prebuilt-layout with markdown output
        poller = client.begin_analyze_document(
            "prebuilt-layout",               # model_id (positional)
            f,                               # body/file-like object (positional)
            content_type="application/pdf",  # important for streams
            output_content_format="markdown"
        )
        result = poller.result()

    # Extract the markdown content
    markdown_content = result.content or ""
    print(f"DEBUG: Result content length: {len(markdown_content)}")
    print(f"DEBUG: Result object: {type(result)}")
    if not markdown_content:
        print("WARNING: No content extracted from PDF")
    return markdown_content


def process_pdf_to_markdown_file(
    pdf_path: str,
    output_path: str,
    endpoint: str,
    key: str
) -> None:
    """
    Process a PDF and save as Markdown file.
    """
    print(f"Processing PDF: {pdf_path}")
    markdown_content = analyze_pdf_to_markdown(pdf_path, endpoint, key)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(markdown_content, encoding='utf-8')
    print(f"Converted {pdf_path} to {output_path}")

if __name__ == "__main__":
    pdf_path = "./resources/B737.pdf"
    output_path = "./resources/B737.md"
    endpoint = "https://aa-genai-train-foundry.cognitiveservices.azure.com/"
    key = "REMOVED_KEY"

    process_pdf_to_markdown_file(
        pdf_path=pdf_path,
        output_path=output_path,
        endpoint=endpoint,
        key=key
    )

