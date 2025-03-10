import re
import pandas as pd
import json
from typing import List, Dict, Union

class MarkdownProcessor:
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200, max_table_size: int = 2000):
        """
        Initialize the Markdown processor.
        
        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks (in characters)
            max_table_size: Maximum size for a table chunk (in characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_size = max_table_size
    
    def parse_markdown(self, markdown_text: str) -> List[Dict[str, Union[str, bool]]]:
        """
        Parse markdown text and identify tables and other content blocks.
        
        Args:
            markdown_text: Raw markdown text
            
        Returns:
            List of dictionaries containing parsed content blocks with metadata
        """
        parsed_blocks = []
        
        # Use a more precise regex pattern that captures entire tables at once
        # This pattern matches a table by finding lines that start with | and end with |
        table_pattern = r'(^\|.*\|[ \t]*$\n(?:^\|.*\|[ \t]*$\n)+)'
        
        # Find all tables in the markdown text
        tables = re.finditer(table_pattern, markdown_text, re.MULTILINE)
        
        # Extract table positions
        table_positions = []
        for match in tables:
            start, end = match.span()
            table_positions.append((start, end, match.group(0)))
        
        # If no tables were found, just return the entire text as a single block
        if not table_positions:
            return [{'content': markdown_text, 'is_table': False, 'position': 0}]
        
        # Process the text, handling tables and non-table content
        last_end = 0
        for start, end, table_content in table_positions:
            # Add any text before this table
            if start > last_end:
                non_table_content = markdown_text[last_end:start]
                if non_table_content.strip():
                    parsed_blocks.append({
                        'content': non_table_content.strip(),
                        'is_table': False,
                        'position': last_end
                    })
            
            # Add the table
            parsed_blocks.append({
                'content': table_content.strip(),
                'is_table': True,
                'position': start,
                'table_data': self._parse_table_structure(table_content)
            })
            
            last_end = end
        
        # Add any remaining text after the last table
        if last_end < len(markdown_text):
            non_table_content = markdown_text[last_end:]
            if non_table_content.strip():
                parsed_blocks.append({
                    'content': non_table_content.strip(),
                    'is_table': False,
                    'position': last_end
                })
        
        return parsed_blocks

    def _parse_table_structure(self, table_markdown: str) -> Dict:
        """
        Parse the structure of a markdown table, extracting headers and rows.
        
        Args:
            table_markdown: Markdown table text
            
        Returns:
            Dictionary containing table structure (headers and rows)
        """
        lines = table_markdown.strip().split('\n')
        
        if len(lines) < 3:
            # Not a valid table (needs header, separator, and at least one data row)
            return {'headers': [], 'rows': []}
        
        # Parse headers (first row)
        header_row = lines[0]
        headers = [h.strip() for h in header_row.split('|')[1:-1]]  # Skip the outer pipes
        
        # Skip separator row (second row)
        
        # Parse data rows
        rows = []
        for i in range(2, len(lines)):
            row = lines[i]
            if row.strip():  # Skip empty lines
                cells = [c.strip() for c in row.split('|')[1:-1]]  # Skip the outer pipes
                # Ensure we have the right number of cells
                if len(cells) < len(headers):
                    cells.extend([''] * (len(headers) - len(cells)))
                elif len(cells) > len(headers):
                    cells = cells[:len(headers)]
                rows.append(cells)
        
        return {
            'headers': headers,
            'rows': rows
        }

    def chunk_text(self, parsed_content: List[Dict]) -> List[Dict]:
        """
        Chunk the parsed content, with special handling for tables.
        
        Args:
            parsed_content: List of content blocks from parse_markdown
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for block in parsed_content:
            if not block['is_table']:
                # Handle regular text using sliding window approach
                text = block['content']
                
                # Split text into sentences for more natural chunks
                sentences = re.split(r'(?<=[.!?])\s+', text)
                current_chunk = ""
                
                for sentence in sentences:
                    # If adding this sentence would exceed chunk size, store current chunk and start a new one
                    if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                        chunks.append({
                            'content': current_chunk.strip(),
                            'is_table': False,
                            'metadata': {
                                'source_type': 'text'
                            }
                        })
                        
                        # Start new chunk with overlap
                        overlap_point = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_point:] + sentence
                    else:
                        current_chunk += sentence + " "
                
                # Add any remaining text as a chunk
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'is_table': False,
                        'metadata': {
                            'source_type': 'text'
                        }
                    })
            else:
                # Handle table content
                table_md = block['content'].strip()
                table_size = len(table_md)
                
                if table_size <= self.max_table_size:
                    chunks.append({
                        'content': table_md,
                        'is_table': True,
                        'metadata': {
                            'source_type': 'table'
                        }
                    })
                else:
                    # Split large tables into smaller chunks
                    rows = block['table_data']['rows']
                    headers = block['table_data']['headers']
                    num_rows = len(rows)
                    
                    # Make sure each row has exactly the same number of columns as headers
                    for i in range(len(rows)):
                        # If a row has too few columns, pad with empty strings
                        if len(rows[i]) < len(headers):
                            rows[i].extend([''] * (len(headers) - len(rows[i])))
                        # If a row has too many columns, truncate
                        elif len(rows[i]) > len(headers):
                            rows[i] = rows[i][:len(headers)]
                    
                    # Process rows in chunks
                    for i in range(0, num_rows, self.chunk_size):
                        chunk_rows = rows[i:i + self.chunk_size]
                        df_chunk = pd.DataFrame(chunk_rows, columns=headers)
                        # Use tablefmt='pipe' for cleaner markdown output with less whitespace
                        table_md_chunk = df_chunk.to_markdown(index=False, tablefmt='pipe')
                        
                        chunks.append({
                            'content': table_md_chunk.strip(),
                            'is_table': True,
                            'metadata': {
                                'source_type': 'table'
                            }
                        })
        
        return chunks

    def save_chunks(self, chunks: List[Dict], file_path: str) -> None:
        """
        Save the chunks to a JSONL file.
        
        Args:
            chunks: List of chunks with metadata
            file_path: Path to the output JSONL file
        """
        with open(file_path, 'w') as file:
            for chunk in chunks:
                file.write(json.dumps(chunk) + '\n')
