from dataclasses import dataclass, field
from typing import List, Optional, Pattern, Set
import re
import unicodedata
from functools import lru_cache


@dataclass
class TextBuilderConfig:
    # Core cleaning
    strip_html: bool = True
    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    strip_control_chars: bool = True
    trim_lines: bool = True
    drop_empty_lines: bool = True

    # Noise removal
    remove_boilerplate: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_marketing: bool = True
    custom_drop_patterns: List[str] = field(default_factory=list)

    # Normalization
    normalize_dates: bool = True
    normalize_units: bool = True
    normalize_numbers: bool = True
    normalize_punctuation: bool = True

    # Language and casing
    target_language: Optional[str] = None  # e.g., 'fr' or 'en'
    lowercase: bool = False
    keep_case_acronyms: bool = True

    # Chunk-prep helpers
    max_line_length: Optional[int] = None  # Changed from 0 to None (more explicit)
    sentence_boundary_hint: bool = True
    preserve_headers: bool = True  # Fixed typo: preserve_header -> preserve_headers

    # Deduplication
    dedup_lines: bool = True
    dedup_chunks: bool = True
    fuzzy_dedup: bool = False

    # Performance
    precompile_regex: bool = True
    fast_html_stripper: bool = True


class TextCleaner:
    """
    High-performance text cleaner for RAG ingestion pipeline.
    Handles HTML, unicode normalization, noise removal, and domain-specific normalization.
    """

    def __init__(self, cfg: TextBuilderConfig):
        self.cfg = cfg
        self._seen_lines: Set[str] = set()  # For dedup
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns for performance"""
        # HTML removal (more robust)
        self.p_html_scripts = re.compile(
            r"<(script|style|noscript|iframe)[\s\S]*?</\1>", re.I
        )
        self.p_html_comments = re.compile(r"<!--[\s\S]*?-->")
        self.p_html_tags = re.compile(r"<[^>]+>")
        
        # Control characters
        self.p_ctrl = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
        self.p_zero_width = re.compile(r"[\u200B-\u200D\uFEFF\u2060]")
        
        # Whitespace normalization
        self.p_horizontal_ws = re.compile(r"[ \t\xa0]+")  # Added non-breaking space
        self.p_vertical_ws = re.compile(r"\n{3,}")
        
        # Noise patterns
        self.p_url = re.compile(
            r"\b(?:https?://|www\.)[^\s<>\"{}|\\^`\[\]]+", re.I
        )
        self.p_email = re.compile(
            r"\b[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )
        self.p_phone = re.compile(
            r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,3}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b"
        )
        
        # Boilerplate (expanded patterns)
        self.p_boilerplate = re.compile(
            r"(?:all rights reserved|©\s*\d{4}|terms of use|cookie policy|"
            r"privacy policy|legal notice|disclaimer|unsubscribe)",
            re.I
        )
        self.p_marketing = re.compile(
            r"(?:subscribe now|newsletter|promo code|discount code|"
            r"sale ends|limited offer|buy now|click here|learn more)",
            re.I
        )
        
        # Domain-specific patterns
        self.p_acronym = re.compile(r"\b([A-Z]{2,})\b")
        
        # Date normalization (multiple formats)
        self.p_date_dd_mm_yyyy = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")
        self.p_date_yyyy_mm_dd = re.compile(r"\b(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})\b")
        
        # Unit normalization (agriculture/science)
        self.p_mm_per_day = re.compile(r"\bmm\s*/\s*(?:day|jour)\b", re.I)
        self.p_l_per_ha = re.compile(r"\b(?:l|liters?|litres?)\s*/\s*(?:ha|hectares?)\b", re.I)
        self.p_m3_per_ha = re.compile(r"\b(?:m3|m³|m\^3)\s*/\s*ha\b", re.I)
        self.p_deg_c = re.compile(r"\b(?:deg(?:rees?)?|°)\s*C(?:elsius)?\b", re.I)
        
        # Number normalization
        self.p_decimal_comma = re.compile(r"(\d+),(\d+)")
        self.p_thin_space_in_numbers = re.compile(r"(?<=\d)[\u2009\u202F\u00A0](?=\d)")
        
        # Punctuation normalization
        self.p_quotes = re.compile(r"[""«»„‟]")
        self.p_apostrophes = re.compile(r"[''`]")
        self.p_dashes = re.compile(r"[–—‒―]")
        self.p_ellipsis = re.compile(r"\.{3,}")
        
        # Custom user patterns
        self.custom_patterns: List[Pattern] = [
            re.compile(p, re.I | re.MULTILINE) for p in self.cfg.custom_drop_patterns
        ]

    def clean(self, text: str) -> str:
        """
        Main cleaning pipeline. Order matters for performance and correctness.
        """
        if not text:
            return ""
        
        s = text
        
        # 1. HTML stripping (early to avoid processing markup)
        if self.cfg.strip_html:
            s = self._strip_html(s)
        
        # 2. Unicode normalization (early for consistent pattern matching)
        if self.cfg.normalize_unicode:
            s = unicodedata.normalize("NFKC", s)
        
        # 3. Control characters removal
        if self.cfg.strip_control_chars:
            s = self.p_ctrl.sub(" ", s)
            s = self.p_zero_width.sub("", s)
        
        # 4. Noise removal (order: specific -> general)
        if self.cfg.remove_boilerplate:
            s = self.p_boilerplate.sub("", s)
        if self.cfg.remove_marketing:
            s = self.p_marketing.sub("", s)
        if self.cfg.remove_urls:
            s = self.p_url.sub(" [URL] ", s)  # Placeholder instead of deletion
        if self.cfg.remove_emails:
            s = self.p_email.sub(" [EMAIL] ", s)
        if self.cfg.remove_phone_numbers:
            s = self.p_phone.sub(" [PHONE] ", s)
        
        # 5. Custom patterns
        for pattern in self.custom_patterns:
            s = pattern.sub(" ", s)
        
        # 6. Domain normalizations (before case changes)
        if self.cfg.normalize_dates:
            s = self._normalize_dates(s)
        if self.cfg.normalize_units:
            s = self._normalize_units(s)
        if self.cfg.normalize_numbers:
            s = self._normalize_numbers(s)
        if self.cfg.normalize_punctuation:
            s = self._normalize_punctuation(s)
        
        # 7. Whitespace normalization
        if self.cfg.collapse_whitespace:
            s = self.p_horizontal_ws.sub(" ", s)
            s = self.p_vertical_ws.sub("\n\n", s)
        
        # 8. Line processing
        if self.cfg.trim_lines:
            s = "\n".join(line.strip() for line in s.splitlines())
        if self.cfg.drop_empty_lines:
            s = "\n".join(line for line in s.splitlines() if line.strip())
        
        # 9. Deduplication
        if self.cfg.dedup_lines:
            s = self._dedup_lines(s)
        
        # 10. Case transformation (late to preserve acronyms)
        if self.cfg.lowercase:
            s = self._apply_lowercase(s)
        
        # 11. Line wrapping (if needed)
        if self.cfg.max_line_length:
            s = self._wrap_lines(s, self.cfg.max_line_length)
        
        return s.strip()

    def _strip_html(self, text: str) -> str:
        """Robust HTML removal"""
        s = self.p_html_scripts.sub(" ", text)
        s = self.p_html_comments.sub(" ", s)
        s = self.p_html_tags.sub(" ", s)
        # Decode common HTML entities
        s = s.replace("&nbsp;", " ")
        s = s.replace("&lt;", "<")
        s = s.replace("&gt;", ">")
        s = s.replace("&amp;", "&")
        s = s.replace("&quot;", '"')
        return s

    def _normalize_dates(self, text: str) -> str:
        """Normalize dates to ISO format YYYY-MM-DD"""
        s = self.p_date_dd_mm_yyyy.sub(r"\3-\2-\1", text)  # DD/MM/YYYY -> YYYY-MM-DD
        s = self.p_date_yyyy_mm_dd.sub(r"\1-\2-\3", s)     # YYYY.MM.DD -> YYYY-MM-DD
        return s

    def _normalize_units(self, text: str) -> str:
        """Normalize scientific/agricultural units"""
        s = self.p_mm_per_day.sub("mm/day", text)
        s = self.p_l_per_ha.sub("L/ha", s)
        s = self.p_m3_per_ha.sub("m³/ha", s)
        s = self.p_deg_c.sub("°C", s)
        return s

    def _normalize_numbers(self, text: str) -> str:
        """Normalize number formatting"""
        s = self.p_thin_space_in_numbers.sub("", text)  # Remove thin spaces
        s = self.p_decimal_comma.sub(r"\1.\2", s)        # 1,5 -> 1.5
        return s

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation to ASCII equivalents"""
        s = self.p_quotes.sub('"', text)
        s = self.p_apostrophes.sub("'", s)
        s = self.p_dashes.sub("-", s)
        s = self.p_ellipsis.sub("...", s)
        return s

    def _dedup_lines(self, text: str) -> str:
        """Remove duplicate consecutive lines"""
        lines = text.splitlines()
        deduped = []
        prev = None
        for line in lines:
            line_stripped = line.strip()
            if line_stripped != prev:
                deduped.append(line)
                prev = line_stripped
        return "\n".join(deduped)

    def _apply_lowercase(self, text: str) -> str:
        """Apply lowercase while preserving acronyms if configured"""
        if self.cfg.keep_case_acronyms:
            # Store acronyms temporarily
            acronyms = {}
            def replace_acronym(match):
                placeholder = f"__ACRONYM{len(acronyms)}__"
                acronyms[placeholder] = match.group(1)
                return placeholder
            
            s = self.p_acronym.sub(replace_acronym, text)
            s = s.lower()
            
            # Restore acronyms
            for placeholder, acronym in acronyms.items():
                s = s.replace(placeholder.lower(), acronym)
            return s
        else:
            return text.lower()

    def _wrap_lines(self, text: str, max_length: int) -> str:
        """Soft-wrap long lines at word boundaries"""
        lines = text.splitlines()
        wrapped = []
        for line in lines:
            if len(line) <= max_length:
                wrapped.append(line)
            else:
                wrapped.extend(self._soft_wrap_line(line, max_length))
        return "\n".join(wrapped)

    @staticmethod
    def _soft_wrap_line(line: str, width: int) -> List[str]:
        """Wrap a single line at word boundaries"""
        if len(line) <= width:
            return [line]
        
        parts = []
        start = 0
        
        while start < len(line):
            end = min(start + width, len(line))
            segment = line[start:end]
            
            # Try to break at last space (but not too early)
            if end < len(line):
                last_space = segment.rfind(" ")
                min_break = int(width * 0.6)  # Don't break too early
                if last_space > min_break:
                    segment = segment[:last_space].rstrip()
                    end = start + last_space + 1  # Skip the space
            
            parts.append(segment)
            start = end
        
        return parts

    def reset_dedup_cache(self) -> None:
        """Clear deduplication cache (call between documents)"""
        self._seen_lines.clear()


# Factory for common configurations
class CleanerPresets:
    @staticmethod
    def rag_ingestion() -> TextBuilderConfig:
        """Preset for RAG document ingestion"""
        return TextBuilderConfig(
            strip_html=True,
            normalize_unicode=True,
            collapse_whitespace=True,
            remove_boilerplate=True,
            remove_urls=False,  # Keep URL markers
            normalize_dates=True,
            normalize_units=True,
            preserve_headers=True,
            dedup_lines=True,
        )

    @staticmethod
    def strict_cleaning() -> TextBuilderConfig:
        """Maximum cleaning for noisy sources"""
        return TextBuilderConfig(
            strip_html=True,
            remove_boilerplate=True,
            remove_marketing=True,
            remove_urls=True,
            remove_emails=True,
            remove_phone_numbers=True,
            normalize_punctuation=True,
            dedup_lines=True,
            fuzzy_dedup=True,
        )

    @staticmethod
    def minimal() -> TextBuilderConfig:
        """Minimal cleaning, preserve structure"""
        return TextBuilderConfig(
            strip_html=False,
            normalize_unicode=True,
            collapse_whitespace=True,
            remove_boilerplate=False,
            normalize_dates=False,
        )


        'i'