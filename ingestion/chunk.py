from pathlib import Path
import json
import re


DATA_DIR = Path("data/rice")
OUTPUT_DIR = Path("data/chunks")

CHUNK_SIZE = 1400
OVERLAP = 200


KNOWN_HEADINGS = {
    "Main Field",
    "Precautions for Irrigation",
    "Alternate Wetting and Drying Irrigation (AWDI)",
    "Field Water Tube - AWDI",
    "Formation of Seedbeds",
    "Nursery - Irrigation Management",
    "Nursery - Weed Management",
    "Seed Treatment",
    "Seed Treatment with Biofertilizers",
    "Seed Treatment with Carrier Based Biofertilizers",
    "Specific gravity grading in rice seeds",
    "Application of Oganic Manures",
    "Green Manure Incorporation",
    "Root Dipping with Carrier Based Biofertilizers",
    "Application of P fertilizer",
    "Foliar Nutrition",
    "Soil Application",
    "P & K may be through Site Specific Nutrient Management (SSNM)",
    "Transplanted Puddled Lowland Rice",
    "Mainfield: Weed management",
    "Special technologies for problem soils",
    "Compact the soil",
    "Pulling Out Seedlings",
}


def clean_line(line):
    """
    Clean formatting noise without changing the actual source wording.
    """
    line = line.replace("\xa0", " ")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def is_table_start(line):
    return line.strip().upper() == "STRUCTURED TABLE DATA"


def detect_sections(text):
    """
    Detect known TNAU section headings while preserving the original
    paragraph/table structure.
    """
    lines = text.splitlines()

    sections = []
    current_heading = "General"
    current_content = []

    for raw_line in lines:
        line = clean_line(raw_line)

        if not line:
            continue

        # Ignore document-level metadata/preamble.
        if line.startswith("Crop Production ::"):
            continue

        if line.endswith(" :: Rice"):
            continue

        if line in KNOWN_HEADINGS:
            if current_content:
                sections.append({
                    "heading": current_heading,
                    "blocks": build_blocks(current_content),
                })

            current_heading = line
            current_content = []

        else:
            current_content.append(line)

    if current_content:
        sections.append({
            "heading": current_heading,
            "blocks": build_blocks(current_content),
        })

    return sections


def build_blocks(lines):
    """
    Convert lines into meaningful blocks.

    A block can be:
    - a paragraph
    - a list-like group
    - a complete structured table

    Tables are kept atomic so their rows are never separated.
    """
    blocks = []
    current = []

    i = 0

    while i < len(lines):
        line = lines[i]

        if is_table_start(line):
            if current:
                blocks.append("\n".join(current))
                current = []

            table_block = [line]
            i += 1

            while i < len(lines):
                table_block.append(lines[i])

                # Tables in our cleaned files continue until the next
                # major non-table content. Keep the complete table block.
                i += 1

                if (
                    i < len(lines)
                    and lines[i] in KNOWN_HEADINGS
                ):
                    break

            blocks.append("\n".join(table_block))
            continue

        current.append(line)

        # Blank lines were already removed by preprocessing, so use
        # sentence endings / source structure as natural boundaries.
        if line.endswith((".", ":", "?")):
            blocks.append("\n".join(current))
            current = []

        i += 1

    if current:
        blocks.append("\n".join(current))

    return blocks


def split_large_block(block):
    """
    Split an unusually large block by words while preserving overlap.
    This is only used when a single logical block exceeds CHUNK_SIZE.
    """
    words = block.split()

    chunks = []
    current_words = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1

        if current_words and current_length + word_length > CHUNK_SIZE:
            chunks.append(" ".join(current_words))

            overlap_words = []
            overlap_length = 0

            for old_word in reversed(current_words):
                addition = len(old_word) + 1

                if overlap_length + addition > OVERLAP:
                    break

                overlap_words.insert(0, old_word)
                overlap_length += addition

            current_words = overlap_words
            current_length = overlap_length

        current_words.append(word)
        current_length += word_length

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def create_chunks(section_heading, blocks):
    """
    Group complete logical blocks into larger semantic chunks.

    Blocks are kept together whenever possible. A block is only split
    internally if it is larger than CHUNK_SIZE.
    """
    chunks = []

    current_blocks = []
    current_length = 0

    section_prefix = f"Section: {section_heading}\n\n"
    prefix_length = len(section_prefix)

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        # Very large logical block: split it independently.
        if len(block) + prefix_length > CHUNK_SIZE:
            if current_blocks:
                chunks.append(
                    section_prefix + "\n\n".join(current_blocks)
                )
                current_blocks = []
                current_length = prefix_length

            large_parts = split_large_block(block)

            for part in large_parts:
                chunks.append(section_prefix + part)

            continue

        block_length = len(block) + 2

        if (
            current_blocks
            and current_length + block_length > CHUNK_SIZE
        ):
            chunks.append(
                section_prefix + "\n\n".join(current_blocks)
            )

            current_blocks = []
            current_length = prefix_length

        current_blocks.append(block)
        current_length += block_length

    if current_blocks:
        chunks.append(
            section_prefix + "\n\n".join(current_blocks)
        )

    return chunks


def main():
    files = sorted(DATA_DIR.glob("*.txt"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    chunk_id = 0

    print(f"Found {len(files)} text files")

    for file in files:
        text = file.read_text(encoding="utf-8")
        sections = detect_sections(text)

        print(f"\n{'=' * 60}")
        print(f"FILE: {file.name}")
        print(f"{'=' * 60}")

        for section in sections:
            section_chunks = create_chunks(
                section["heading"],
                section["blocks"],
            )

            print(
                f"[{section['heading']}] "
                f"→ {len(section_chunks)} chunk(s)"
            )

            for chunk in section_chunks:
                all_chunks.append({
                    "chunk_id": f"rice_{chunk_id:04d}",
                    "source": file.name,
                    "crop": "rice",
                    "section": section["heading"],
                    "text": chunk,
                })

                chunk_id += 1

    output_file = OUTPUT_DIR / "rice_chunks.jsonl"

    with output_file.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                ) + "\n"
            )

    print(f"\n{'=' * 60}")
    print("CHUNKING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
