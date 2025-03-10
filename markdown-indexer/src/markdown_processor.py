import re
import pandas as pd
from typing import List, Dict, Union

class MarkdownProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, max_table_size: int = 2000):
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
        
        # Regular expression to identify markdown tables
        # This pattern looks for lines starting with | and containing at least one more |
        table_pattern = r'(^\|.*?\|.*$\n)((?:^\|.*?\|.*$\n)+)'
        
        # Split the text at table boundaries
        split_text = re.split(f'({table_pattern})', markdown_text, flags=re.MULTILINE)
        
        current_position = 0
        i = 0
        while i < len(split_text):
            block = split_text[i].strip()
            
            # If this is a table match (starts with | and contains | character)
            if block and re.match(r'^\|.*\|.*$', block.split('\n')[0]):
                # It's a table - might be just the first line of the table
                table_content = block
                
                # If next parts exist and they're part of the table (regex capture groups)
                if i + 2 < len(split_text) and split_text[i+1].strip():
                    # Include the captured groups
                    table_content = table_content + split_text[i+1].strip()
                    i += 1
                
                # Parse table structure if possible
                table_data = self._parse_table_structure(table_content)
                
                parsed_blocks.append({
                    'content': table_content,
                    'is_table': True,
                    'position': current_position,
                    'table_data': table_data
                })
                
                current_position += len(table_content)
                
            elif block:  # Only process non-empty blocks
                # It's regular text
                parsed_blocks.append({
                    'content': block,
                    'is_table': False,
                    'position': current_position
                })
                
                current_position += len(block)
            
            i += 1
        
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
                table_md = block['content']
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
                    chunk_rows = []
                    
                    for i in range(0, num_rows, self.chunk_size):
                        chunk_rows = rows[i:i + self.chunk_size]
                        df_chunk = pd.DataFrame(chunk_rows, columns=headers)
                        table_md_chunk = df_chunk.to_markdown(index=False)
                        
                        chunks.append({
                            'content': table_md_chunk,
                            'is_table': True,
                            'metadata': {
                                'source_type': 'table'
                            }
                        })
        
        return chunks
