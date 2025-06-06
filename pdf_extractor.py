"""
Module: pdf_extractor.py
Provides a unified PDF extraction facade using Tabula, pdfplumber, and Camelot.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import tabula
import pdfplumber
import camelot

from racing_system import AdvancedPDFExtractor

class PDFExtractor:
    """
    Facade for extracting race info and horse tables from PDF,
    trying Tabula, then pdfplumber, then Camelot.
    """
    def __init__(self, db, pdf_config: Dict, logger: logging.Logger):
        # Initialize underlying extractor
        self._orig = AdvancedPDFExtractor(db, pdf_config, logger)
        self.logger = logger

    def extract_race_info(self, pdf_path: Path, force_refresh: bool = False) -> Any:
        # Delegate to original for full flow (cache, text extract)
        return self._orig.extract_race_info(pdf_path, force_refresh)

    def extract_horses_from_pdf(self, pdf_path: Path, race_info: Dict, text: str) -> Any:
        # Try Tabula
        try:
            dfs = tabula.read_pdf(str(pdf_path), **self._orig.tabula_options[0])
            if dfs:
                horses = self._orig._parse_horses_from_tables(dfs, race_info)
                if horses:
                    return horses
        except Exception as e:
            self.logger.warning(f"Tabula extraction failed: {e}")
        # Try pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                tables = []
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables.append(df)
                if tables:
                    horses = self._orig._parse_horses_from_tables(tables, race_info)
                    if horses:
                        return horses
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
        # Try Camelot
        try:
            tables = camelot.read_pdf(str(pdf_path), pages='all')
            dfs = [t.df for t in tables]
            if dfs:
                horses = self._orig._parse_horses_from_tables(dfs, race_info)
                return horses
        except Exception as e:
            self.logger.warning(f"Camelot extraction failed: {e}")

    # Add validation delegation to facade
    def validate_extracted_data(self, info: Any) -> Any:
        """Delegate validation to AdvancedPDFExtractor."""
        return self._orig.validate_extracted_data(info)