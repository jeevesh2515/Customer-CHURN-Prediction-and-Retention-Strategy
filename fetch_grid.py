import requests
from bs4 import BeautifulSoup

def fetch_and_print_grid(doc_url):
    """
    Fetches a Google Doc from the given URL, parses the data for Unicode characters
    and their x/y coordinates, and prints the grid representing the message.
    """
    # Fetch the document content
    response = requests.get(doc_url)
    if response.status_code != 200:
        raise ValueError(f"Unable to fetch document. Status code: {response.status_code}")
    
    # Parse the content using BeautifulSoup (assuming it's in an HTML format)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract the table rows (assuming the document contains a single table)
    table_rows = soup.find_all('tr')
    grid_data = []

    # Parse the rows and collect coordinates and characters
    for row in table_rows[1:]:  # Skip the header row
        cells = row.find_all('td')
        if len(cells) != 3:  # Expect exactly 3 columns per row
            continue
        x = int(cells[0].text.strip())
        character = cells[1].text.strip()
        y = int(cells[2].text.strip())
        grid_data.append((x, y, character))
    
    # Determine the grid dimensions
    max_x = max(x for x, _, _ in grid_data)
    max_y = max(y for _, y, _ in grid_data)
    
    # Initialise an empty grid with spaces
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    
    # Populate the grid with characters
    for x, y, character in grid_data:
        grid[max_y - y][x] = character
    
    # Print the grid
    for row in grid:
        print(''.join(row))

# Example usage
doc_url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"
fetch_and_print_grid(doc_url)