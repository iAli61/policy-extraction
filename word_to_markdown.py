#!/usr/bin/env python3
"""
Word to Markdown Converter

This script converts Microsoft Word documents (.docx) to Markdown format (.md)
while preserving the original document's layout, tables, and images.

Dependencies:
- python-docx (for reading Word documents)
- mammoth (for main conversion)
- beautifulsoup4 (for HTML parsing)
- markdownify (for HTML to Markdown conversion)
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
        p[style-name='heading 1'] => h1:fresh
        p[style-name='heading 2'] => h2:fresh
        p[style-name='heading 3'] => h3:fresh
        p[style-name='heading 4'] => h4:fresh
        p[style-name='heading 5'] => h5:fresh
        p[style-name='heading 6'] => h6:fresh
        p[style-name='title'] => h1:fresh
        p[style-name='subtitle'] => h2:fresh
        p[style-name='quote'] => blockquote
        r[style-name='strong'] => strong
        r[style-name='emphasis'] => em
        p[style-name='list paragraph'] => ul > li:fresh
        """
    
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
            
            # Merge adjacent tables that appear to be continuations
            next_sibling = table.find_next_sibling()
            if next_sibling and next_sibling.name == 'table':
                # Check if this might be a continuation (rough heuristic)
                if len(table.find_all('tr')) < 20:  # Not already a very large table
                    # Move all rows from the next table into this one
                    for row in next_sibling.find_all('tr'):
                        table.append(row)
                    # Remove the next table since we've merged it
                    next_sibling.extract()
            
            # Ensure all cells have aligned content
            for row in table.find_all('tr'):
                for cell in row.find_all(['td', 'th']):
                    cell['style'] = "vertical-align: top;"
                    
        return str(soup)
    
    def _html_to_markdown(self, html_content):
        """Convert HTML to Markdown using markdownify with custom options."""
        # Before conversion, ensure image tags are properly formatted
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Fix image tags to ensure they convert properly to markdown
        for img in soup.find_all('img'):
            # Make sure alt text is present
            if not img.get('alt'):
                img['alt'] = "Image"
            
            # Ensure src is properly formatted
            if img.get('src') and not img['src'].startswith(('http', 'https', '/')):
                # For relative paths, ensure they're formatted correctly
                img['src'] = img['src'].replace('\\', '/')
        
        # Remove empty paragraphs that might cause breaks in tables
        for p in soup.find_all('p'):
            if not p.get_text(strip=True) and not p.find_all(['img', 'br']):
                p.extract()
        
        html_content = str(soup)
        
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
            
            # Remove any lines that just have | and whitespace (broken table rows)
            table_lines = table_text.split('\n')
            cleaned_lines = []
            for line in table_lines:
                if line.strip() and not re.match(r'^\s*\|\s*$', line):
                    cleaned_lines.append(line)
            
            # Join back together
            cleaned_table = '\n'.join(cleaned_lines)
            
            # Ensure the table has a header separator row
            lines = cleaned_table.strip().split('\n')
            if len(lines) >= 2 and '|--' not in lines[1]:
                header = lines[0]
                col_count = header.count('|') - 1
                separator = '|' + '---|' * col_count
                lines.insert(1, separator)
                new_table = '\n'.join(lines)
                markdown = markdown.replace(table_text, new_table + '\n\n')
            else:
                # Replace with cleaned version anyway
                markdown = markdown.replace(table_text, cleaned_table + '\n\n')
        
        # Fix broken table continuations (where a table is split with an empty line)
        markdown = re.sub(r'(\|\s*\n\s*\n\s*\|)', '|\n|', markdown)
        
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
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Create images directory for storing extracted images
        images_dir = output_path.parent / "images"
        os.makedirs(images_dir, exist_ok=True)
        
        # Image handler for mammoth
        def handle_image(image):
            with image.open() as image_data:
                content_bytes = image_data.read()
                
            # Generate a unique filename for the image
            image_filename = f"image_{hash(content_bytes) % 10000}.{image.content_type.split('/')[-1]}"
            image_path = images_dir / image_filename
            
            # Save the image
            with open(image_path, "wb") as f:
                f.write(content_bytes)
                
            print(f"Saved image: {image_path}")
            
            # Return the image reference for the markdown
            return {
                "src": f"images/{image_filename}",
                "alt_text": image.alt_text or "Image"
            }
        
        # Options for the conversion
        options = {
            "style_map": self.style_map
        }
        
        # Add image handling if requested
        if self.preserve_images:
            options["convert_image"] = mammoth.images.img_element(lambda image: {
                "src": handle_image(image)["src"],
                "alt": handle_image(image)["alt_text"]
            })
        
        # Convert Word to HTML using mammoth
        with open(input_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file, **options)
            html = result.value
            
            # Log any messages
            for message in result.messages:
                print(f"Warning: {message}")
        
        # Fix HTML for better markdown conversion
        if self.preserve_tables:
            html = self._fix_tables(html)
        
        # Convert HTML to Markdown
        markdown = self._html_to_markdown(html)
        
        # Write the Markdown content to the output file
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
        description="Convert Word documents to Markdown while preserving layout, tables, and images."
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