import requests
from bs4 import BeautifulSoup
from pathlib import Path


def extract_tables(soup):
    """
    Extract meaningful data tables from an HTML page.

    Ignores layout/navigation tables and preserves
    actual rows and columns.
    """

    tables = []

    for table in soup.find_all("table"):

        rows = []

        for tr in table.find_all("tr"):

            cells = tr.find_all(["th", "td"])

            row = []

            for cell in cells:

                text = cell.get_text(" ", strip=True)

                if text:
                    row.append(text)

            if row:
                rows.append(row)

        # Ignore empty tables
        if not rows:
            continue

        # ---------------------------------------------
        # Ignore obvious navigation/footer tables
        # ---------------------------------------------

        combined_text = " ".join(
            " ".join(row)
            for row in rows
        ).lower()

        navigation_words = [
            "home",
            "seasons & varieties",
            "tillage",
            "nutrient management",
            "irrigation management",
            "weed management",
            "crop protection",
            "cost of cultivation",
            "© all rights reserved"
        ]

        navigation_matches = sum(
            1
            for word in navigation_words
            if word in combined_text
        )

        if navigation_matches >= 3:
            continue

        # Ignore tiny tables such as image labels
        if len(rows) == 1 and len(rows[0]) <= 2:
            continue

        # ---------------------------------------------
        # Normalize column count
        # ---------------------------------------------

        max_columns = max(
            len(row)
            for row in rows
        )

        normalized_rows = []

        for row in rows:

            row = row + [""] * (
                max_columns - len(row)
            )

            normalized_rows.append(row)

        # ---------------------------------------------
        # Convert to pipe-separated format
        # ---------------------------------------------

        table_lines = []

        for row in normalized_rows:

            table_lines.append(
                " | ".join(row)
            )

        tables.append(table_lines)

    return tables


def extract_text(url, output_file):

    print("\nDownloading:")
    print(url)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "lxml"
    )

    # --------------------------------------------------
    # REMOVE UNWANTED HTML
    # --------------------------------------------------

    for element in soup([
        "script",
        "style",
        "noscript"
    ]):
        element.decompose()

    # --------------------------------------------------
    # EXTRACT TABLES
    # --------------------------------------------------

    tables = extract_tables(soup)

    # --------------------------------------------------
    # NORMAL TEXT
    # --------------------------------------------------
    # (The line that removed tables has been deleted,
    #  so tables remain in the soup and will appear
    #  as plain text below.)

    text = soup.get_text(separator="\n")

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    # --------------------------------------------------
    # REMOVE TNAU NAVIGATION
    # --------------------------------------------------

    navigation_items = {
        "Home |",
        "Home",
        "Seasons & Varieties |",
        "Seasons & Varieties",
        "Tillage |",
        "Tillage",
        "Nutrient Mgmnt |",
        "Nutrient Mgmnt",
        "Nutrient Management",
        "Irrigation Mgmnt |",
        "Irrigation Mgmnt",
        "Irrigation Management",
        "Weed Mgmnt |",
        "Weed Mgmnt",
        "Weed Management",
        "Crop Protection |",
        "Crop Protection",
        "Cost of Cultivation",
        "Photobank",
        "|"
    }

    clean_lines = []

    for line in lines:

        if line in navigation_items:
            continue

        clean_lines.append(line)

    # --------------------------------------------------
    # REMOVE IMAGE SOURCE BLOCKS
    # --------------------------------------------------

    filtered_lines = []

    skip_image_source = False

    for line in clean_lines:

        if line.lower().startswith("image source:"):
            skip_image_source = True
            continue

        if skip_image_source:

            if (
                "www." in line.lower()
                or "http" in line.lower()
                or "professor" in line.lower()
                or "tnau" in line.lower()
                or "coimbatore" in line.lower()
            ):
                continue

            skip_image_source = False

        filtered_lines.append(line)

    clean_lines = filtered_lines

    # --------------------------------------------------
    # REMOVE DUPLICATE CONSECUTIVE LINES
    # --------------------------------------------------

    deduplicated_lines = []

    for line in clean_lines:

        if (
            not deduplicated_lines
            or line != deduplicated_lines[-1]
        ):
            deduplicated_lines.append(line)

    clean_lines = deduplicated_lines

    # --------------------------------------------------
    # FIX BROKEN COLONS
    # --------------------------------------------------

    formatted_lines = []

    i = 0

    while i < len(clean_lines):

        current = clean_lines[i]

        if (
            i + 2 < len(clean_lines)
            and clean_lines[i + 1] == ":"
        ):

            combined = (
                current
                + ": "
                + clean_lines[i + 2]
            )

            formatted_lines.append(combined)

            i += 3

        else:

            formatted_lines.append(current)

            i += 1

    clean_lines = formatted_lines

    # --------------------------------------------------
    # REMOVE COPYRIGHT
    # --------------------------------------------------

    final_lines = []

    for line in clean_lines:

        if line.startswith(
            "© All Rights Reserved"
        ):
            continue

        final_lines.append(line)

    # --------------------------------------------------
    # ADD STRUCTURED TABLE DATA
    # --------------------------------------------------

    if tables:

        final_lines.append("")
        final_lines.append(
            "STRUCTURED TABLE DATA"
        )

        for index, table in enumerate(
            tables,
            start=1
        ):

            final_lines.append("")
            final_lines.append(
                f"TABLE {index}"
            )

            for row in table:
                final_lines.append(row)

    # --------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------

    clean_text = "\n".join(final_lines)

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        clean_text,
        encoding="utf-8"
    )

    print(f"Saved → {output_path}")
    print(f"Characters: {len(clean_text)}")
    print(f"Lines: {len(final_lines)}")
    print(f"Tables: {len(tables)}")
