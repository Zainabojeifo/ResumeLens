"""
STEP 2 – DATA CLEANING
Text from PDFs, Word files and OCR is messy in different ways. I clean it so that
every resume ends up in the same tidy shape before the NER model sees it:

  * fix PDF glyph codes like "(cid:127)" and odd characters (ligatures, curly quotes)
  * turn every kind of bullet (•, *, «, », e) into "- " at the start of a line
  * turn separators between items on one line (•, ·, «, *) into " | "
  * use one kind of dash, remove stray symbols, squeeze repeated spaces
  * remove blank lines and re-join lines that were broken in the middle of a sentence
  * find the sections of the resume (Summary, Experience, Education, Skills)

Important design choice: I apply exactly the same cleaning to the training data.
Because cleaning changes the length of the text, `clean_text` keeps an offset map,
so the labelled entity positions can be moved to the right place in the cleaned text.
"""
import bisect
import re

# (pattern, replacement, flags) applied in order
CLEANING_RULES = [
    (r"\(cid:\d+\)", "•", 0),                      # PDF glyph codes (usually bullets)
    (r"ﬁ", "fi", 0), (r"ﬂ", "fl", 0),              # ligatures
    (r"[“”]", '"', 0), (r"[‘’`]", "'", 0),         # curly quotes
    (r"[–—−]", "-", 0),                            # all dashes -> "-"
    (r"\r", "", 0),
    (r"[ \t ]+", " ", 0),                     # repeated spaces / tabs
    (r"^ +| +$", "", re.M),                        # spaces at line start / end
    # bullets at the start of a line (OCR often reads "•" as e, o, «, », ¢ or °)
    (r"^(?:[•·*«»¢°■▪➢✓>-]|[eo](?= [A-Z]))\s*", "- ", re.M),
    # separators between items on the same line
    (r" (?:[•·«»*]|\.) (?=\S)", " | ", 0),
    (r"[^\w\s@.,;:()/&|+#%'\"!?$-]", "", 0),        # stray symbols
    (r"\n{2,}", "\n", 0),                           # blank lines
    (r"(?<=[a-z,])\n(?=[a-z]+[ ,])", " ", 0),      # a sentence broken over two lines
    (r"^\n+|\n+$", "", 0),
]

SECTION_KEYWORDS = {
    "summary": ["summary", "profile", "objective", "about me"],
    "experience": ["experience", "work history", "employment", "career history"],
    "education": ["education", "academic", "qualifications"],
    "skills": ["skills", "tools", "competencies", "technologies"],
}


def _substitute(pattern, repl, text, offsets, flags=0):
    """re.sub that also updates the offset map.

    offsets[i] = position in the ORIGINAL text of character i of `text`
    (plus one extra item at the end for the end of the text).
    """
    out, new_offsets, pos = [], [], 0
    for m in re.finditer(pattern, text, flags):
        if m.end() == m.start():
            continue
        out.append(text[pos:m.start()])
        new_offsets.extend(offsets[pos:m.start()])
        replacement = m.expand(repl)
        out.append(replacement)
        length = m.end() - m.start()
        new_offsets.extend(offsets[m.start() + min(k, length - 1)] for k in range(len(replacement)))
        pos = m.end()
    out.append(text[pos:])
    new_offsets.extend(offsets[pos:])
    return "".join(out), new_offsets


def clean_text(text, return_offsets=False):
    """Clean raw resume text. Optionally return the offset map (for labelled data)."""
    offsets = list(range(len(text) + 1))
    for pattern, repl, flags in CLEANING_RULES:
        text, offsets = _substitute(pattern, repl, text, offsets, flags)
    return (text, offsets) if return_offsets else text


def move_spans(entities, offsets, cleaned_text):
    """Move labelled (start, end, label) spans from the raw text to the cleaned text."""
    moved = []
    for start, end, label in entities:
        new_start = bisect.bisect_left(offsets, start)
        new_end = bisect.bisect_left(offsets, end)
        # trim spaces that may sit on the edge of the span after cleaning
        while new_start < new_end and cleaned_text[new_start].isspace():
            new_start += 1
        while new_end > new_start and cleaned_text[new_end - 1].isspace():
            new_end -= 1
        if new_end > new_start:
            moved.append((new_start, new_end, label))
    return moved


def is_heading(line):
    """A heading is a short line (1-4 words) containing a section keyword."""
    words = line.strip().rstrip(":").split()
    if not 1 <= len(words) <= 4:
        return None
    low = " ".join(words).lower()
    for section, keys in SECTION_KEYWORDS.items():
        if any(k in low for k in keys):
            return section
    return None


def find_sections(text):
    """Return a list of (section, start, end) character spans. Text before the
    first heading is the 'header' (name and contact details)."""
    sections, current, start, pos = [], "header", 0, 0
    for line in text.splitlines(keepends=True):
        section = is_heading(line)
        if section:
            sections.append((current, start, pos))
            current, start = section, pos + len(line)
        pos += len(line)
    sections.append((current, start, len(text)))
    return [s for s in sections if s[2] > s[1]]


def section_at(sections, char_pos):
    for name, start, end in sections:
        if start <= char_pos < end:
            return name
    return "header"


if __name__ == "__main__":
    import sys
    from step1_extract import extract_text
    raw = extract_text(sys.argv[1])
    cleaned = clean_text(raw)
    print(cleaned)
    print("\nSections:")
    for name, s, e in find_sections(cleaned):
        print(f"  {name:10s} chars {s}-{e}")
