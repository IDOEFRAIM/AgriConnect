# document_loader.py
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any

class DocumentLoader:
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)

    def _read_json_file(self, file: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            records.append(item)
                        else:
                            records.append({"value": item})
                elif isinstance(data, dict):
                    records.append(data)
                else:
                    records.append({"value": data})
        except Exception as e:
            print(f"[JSON] Erreur lecture {file.name} → {e}")
        return records

    def _read_jsonl_file(self, file: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        try:
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj)
                        else:
                            records.append({"value": obj})
                    except Exception:
                        continue
        except Exception as e:
            print(f"[JSONL] Erreur lecture {file.name} → {e}")
        return records

    def _read_csv_file(self, file: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        try:
            with file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))
        except Exception as e:
            print(f"[CSV] Erreur lecture {file.name} → {e}")
        return records

    def _read_txt_file(self, file: Path) -> List[Dict[str, Any]]:
        try:
            text = file.read_text(encoding="utf-8")
            return [{"filename": file.name, "content": text}]
        except Exception as e:
            print(f"[TXT] Erreur lecture {file.name} → {e}")
            return []

    def load_all(self, with_filenames: bool = False, recursive: bool = False) -> Union[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]]]:
        """
        Charge tous les formats supportés dans le dossier.

        Args:
            with_filenames: si True, retourne une liste de tuples (document, filename)
            recursive: si True, parcourt les sous-dossiers

        Returns:
            Liste de documents (ou liste de tuples (doc, filename) si with_filenames=True)
        """
        pattern = "**/*" if recursive else "*"
        json_files = list(self.data_dir.glob(f"{pattern}.json"))
        jsonl_files = list(self.data_dir.glob(f"{pattern}.jsonl"))
        csv_files = list(self.data_dir.glob(f"{pattern}.csv"))
        txt_files = list(self.data_dir.glob(f"{pattern}.txt"))

        results: List[Union[Dict[str, Any], Tuple[Dict[str, Any], str]]] = []

        for file in json_files:
            docs = self._read_json_file(file)
            for d in docs:
                if with_filenames:
                    results.append((d, file.name))
                else:
                    results.append(d)

        for file in jsonl_files:
            docs = self._read_jsonl_file(file)
            for d in docs:
                if with_filenames:
                    results.append((d, file.name))
                else:
                    results.append(d)

        for file in csv_files:
            docs = self._read_csv_file(file)
            for d in docs:
                if with_filenames:
                    results.append((d, file.name))
                else:
                    results.append(d)

        for file in txt_files:
            docs = self._read_txt_file(file)
            for d in docs:
                if with_filenames:
                    results.append((d, file.name))
                else:
                    results.append(d)

        return results