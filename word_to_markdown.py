#!/usr/bin/env python3
"""
Word to Markdown Converter

This script converts Microsoft Word documents (.docx) to Markdown format (.md)
while preserving the original document's layout and tables.

Dependencies:
- python-docx (for reading Word documents)
- mammoth (for main conversion)
- beautifulsoup4 (for HTML parsing)
- markdown-table-formatter (for table beautification)
"""

import os
import re
import sys
import argparse
from pathlib import Path

try:
    import docx
    import mammoth
    from bs4 import BeautifulSoup
    import markdownify
except ImportError:
    print("Installing required dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "python-docx", "mammoth", "beautifulsoup4", "markdownify"])
    import docx
    import mammoth
    from bs4 import BeautifulSoup
    import markdownify

class WordToMarkdownConverter:
    """Converts Word documents to Markdown with layout preservation."""
    
    def __init__(self, preserve_tables=True, preserve_images=True, preserve_lists=True):
        """Initialize the converter with preservation options."""
        self.preserve_tables = preserve_tables
        self.preserve_images = preserve_images
        self.preserve_lists = preserve_lists
        
        # Custom style map to better handle Word formatting
        self.style_map = """
        p[style-name='Heading 1'] => # {}
        p[style-name='Heading 2'] => ## {}
        p[style-name='Heading 3'] => ### {}
        p[style-name='Heading 4'] => #### {}
        p[style-name='Heading 5'] => ##### {}
        p[style-name='Heading 6'] => ###### {}
        p[style-name='Title'] => # {}
        p[style-name='Subtitle'] => ## {}
        p[style-name='Quote'] => > {}
        r[style-name='Strong'] => **{}**
        r[style-name='Emphasis'] => *{}*
        """
        
    def _extract_image_data(self, doc_path):
        """Extract images from the Word document for later reference."""
        doc = docx.Document(doc_path)
        image_data = []
        
        # Get all parts of the document
        for rel in doc.part.rels.values():
            # Check if it's an image
            if "image" in rel.target_ref:
                try:
                    image_type = rel.target_ref.split('.')[-1]
                    image_data.append({
                        'id': rel.rId,
                        'type': image_type,
                        'data': rel.target_part.blob
                    })
                except Exception as e:
                    print(f"Warning: Could not extract image: {e}")
                    
        return image_data
    
    def _fix_tables(self, html_content):
        """Improve the formatting of tables in the HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        for table in tables:
            # Add borders and proper spacing to tables
            table['border'] = "1"
            table['cellpadding'] = "5"
            table['cellspacing'] = "0"
            
            # Ensure header row is properly formatted
            if table.thead:
                headers = table.thead.find_all('th')
                for header in headers:
                    if header.string:
                        header.string.wrap(soup.new_tag('strong'))
                        
            # Ensure all cells have aligned content
            for row in table.find_all('tr'):
                for cell in row.find_all(['td', 'th']):
                    cell['style'] = "vertical-align: top;"
                    
        return str(soup)
    
    def _html_to_markdown(self, html_content):
        """Convert HTML to Markdown using markdownify with custom options."""
        converter = markdownify.MarkdownConverter(
            heading_style="atx",  # Use # style headings
            bullets="-",          # Use - for unordered lists
            strip=["script", "style"],
            escape_asterisks=False,
            escape_underscores=False
        )
        
        markdown = converter.convert(html_content)
        
        # Additional clean-up for better Markdown formatting
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)  # Remove excess newlines
        
        # Fix table formatting for better markdown compatibility
        table_pattern = r'(\|.*\|\n)+'
        tables = re.finditer(table_pattern, markdown)
        
        for table_match in tables:
            table_text = table_match.group(0)
            
            # Ensure the table has a header separator row
            lines = table_text.strip().split('\n')
            if len(lines) >= 2 and '|--' not in lines[1]:
                header = lines[0]
                col_count = header.count('|') - 1
                separator = '|' + '---|' * col_count
                lines.insert(1, separator)
                new_table = '\n'.join(lines)
                markdown = markdown.replace(table_text, new_table + '\n\n')
        
        return markdown
    
    def convert_file(self, input_path, output_path=None):
        """
        Convert a Word file to Markdown.
        
        Args:
            input_path: Path to the Word document
            output_path: Optional path for the output Markdown file.
                        If not provided, uses the same name with .md extension.
        
        Returns:
            Path to the generated markdown file
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        if output_path is None:
            output_path = input_path.with_suffix('.md')
        else:
            output_path = Path(output_path)
        
        # Step 1: Extract images if needed
        if self.preserve_images:
            image_data = self._extract_image_data(input_path)
        
        # Step 2: Convert Word to HTML using mammoth
        with open(input_path, "rb") as docx_file:
            result = mammoth.convert_to_html(
                docx_file, 
                style_map=self.style_map
            )
            html = result.value
            
            # Log any messages (warnings)
            for message in result.messages:
                print(f"Warning: {message}")
        
        # Step 3: Fix HTML for better markdown conversion
        if self.preserve_tables:
            html = self._fix_tables(html)
        
        # Step 4: Convert HTML to Markdown
        markdown = self._html_to_markdown(html)
        
        # Step 5: Write the Markdown content to the output file
        with open(output_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown)
        
        print(f"Converted: {input_path} → {output_path}")
        return output_path
    
    def convert_directory(self, input_dir, output_dir=None, recursive=False):
        """
        Convert all Word documents in a directory to Markdown.
        
        Args:
            input_dir: Path to the directory containing Word documents
            output_dir: Optional path for the output directory
            recursive: Whether to process subdirectories
            
        Returns:
            List of paths to generated markdown files
        """
        input_dir = Path(input_dir)
        
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")
        
        if output_dir is None:
            output_dir = input_dir
        else:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        
        converted_files = []
        
        # Get all Word documents
        pattern = "**/*.docx" if recursive else "*.docx"
        for docx_path in input_dir.glob(pattern):
            rel_path = docx_path.relative_to(input_dir)
            md_path = output_dir / rel_path.with_suffix('.md')
            
            # Create subdirectories if needed
            os.makedirs(md_path.parent, exist_ok=True)
            
            # Convert the file
            converted_file = self.convert_file(docx_path, md_path)
            converted_files.append(converted_file)
        
        return converted_files

def main():
    """Main function to handle command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert Word documents to Markdown while preserving layout and tables."
    )
    
    parser.add_argument("input", help="Input Word document or directory")
    parser.add_argument("-o", "--output", help="Output Markdown file or directory")
    parser.add_argument("-r", "--recursive", action="store_true", 
                        help="Process directories recursively")
    parser.add_argument("--no-tables", action="store_true", 
                        help="Don't attempt to preserve tables")
    parser.add_argument("--no-images", action="store_true", 
                        help="Don't attempt to preserve images")
    parser.add_argument("--no-lists", action="store_true", 
                        help="Don't attempt to preserve lists")
    
    args = parser.parse_args()
    
    converter = WordToMarkdownConverter(
        preserve_tables=not args.no_tables,
        preserve_images=not args.no_images,
        preserve_lists=not args.no_lists
    )
    
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            converter.convert_file(input_path, args.output)
        elif input_path.is_dir():
            converter.convert_directory(input_path, args.output, args.recursive)
        else:
            print(f"Error: {input_path} is not a valid file or directory")
            return 1
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())