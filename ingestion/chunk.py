from pathlib import Path

DATA_DIR = Path("data/rice")
OUTPUT_DIR = Path("data/chunks")

CHUNK_SIZE = 900
OVERLAP = 100

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


def detect_sections(text):
    lines = text.splitlines()

    sections = []
    current_heading = "General"
    current_content = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line in KNOWN_HEADINGS:
            if current_content:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_content)
                })

            current_heading = line
            current_content = []

        else:
            current_content.append(line)

    if current_content:
        sections.append({
            "heading": current_heading,
            "text": "\n".join(current_content)
        })

    return sections


def create_chunks(text):
    words = text.split()

    chunks = []
    current_words = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1

        if current_length + word_length > CHUNK_SIZE:
            chunks.append(" ".join(current_words))

            overlap_words = []
            overlap_length = 0

            for old_word in reversed(current_words):
                if overlap_length + len(old_word) + 1 > OVERLAP:
                    break

                overlap_words.insert(0, old_word)
                overlap_length += len(old_word) + 1

            current_words = overlap_words
            current_length = overlap_length

        current_words.append(word)
        current_length += word_length

    if current_words:
        chunks.append(" ".join(current_words))

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
            section_chunks = create_chunks(section["text"])

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
                    "text": chunk
                })

                chunk_id += 1

    output_file = OUTPUT_DIR / "rice_chunks.jsonl"

    with output_file.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            import json
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print("CHUNKING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
