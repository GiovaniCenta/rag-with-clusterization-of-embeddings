import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Frame
from xml.sax.saxutils import escape

def create_pdf(data, filename):
    # Set up the PDF canvas
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4  # A4 dimensions
    styles = getSampleStyleSheet()
    style = styles['BodyText']

    max_page = data['page'].max()
    for page_number in range(1, max_page + 1):
        page_data = data[data['page'] == page_number]
        if not page_data.empty:
            # Escape HTML content in the text to prevent formatting errors
            text = escape(page_data.iloc[0]['chunk_text'])
            # Create a Paragraph object, which handles text wrapping automatically
            paragraph = Paragraph(text, style)
            # Draw the paragraph on the canvas using a temporary Frame
            frame = Frame(72, 72, width - 144, height - 144, leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
            frame.addFromList([paragraph], c)
        if page_number < max_page or not page_data.empty:
            c.showPage()  # Ensure a new page is started only if there's more content

    c.save()

# Load your JSON data
df = pd.read_json('datasets/dataset.json')  # Adjust the filename to your JSON file

# Sort data by 'title' and 'page' to maintain order in the PDF
df.sort_values(by=['title', 'page'], inplace=True)

# Group by 'title' and process each group into a separate PDF
for title, group in df.groupby('title'):
    filename = f"datasets/{title.replace(' ', '_').lower()}.pdf"
    create_pdf(group, filename)
    print(f"Created PDF for {title}")
