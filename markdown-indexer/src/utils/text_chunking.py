def chunk_text(markdown_text, max_chunk_size=500):
    import re

    # Split the markdown text into lines
    lines = markdown_text.splitlines()
    chunks = []
    current_chunk = []

    for line in lines:
        # Check if the line is a table
        if re.match(r'^\|.*\|$', line):  # Simple regex to identify table rows
            # If current chunk is empty, add the entire table
            if not current_chunk:
                current_chunk.append(line)
            else:
                # Check if adding this line exceeds the max chunk size
                if sum(len(l) for l in current_chunk) + len(line) + len(current_chunk) > max_chunk_size:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]  # Start a new chunk with the current table row
                else:
                    current_chunk.append(line)
        else:
            # If it's not a table line, check if it can be added to the current chunk
            if sum(len(l) for l in current_chunk) + len(line) + len(current_chunk) > max_chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]  # Start a new chunk with the current line
            else:
                current_chunk.append(line)

    # Add any remaining lines as a chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks