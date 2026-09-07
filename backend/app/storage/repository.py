"""
Storage Repository for Inspection Claims.
Provides an abstract repository pattern with a JSON-backed local implementation for Phase 1,
designed for seamless plug-in replacement with MongoDB in Phase 2.
"""

from abc import ABC, abstractmethod
import json
import os
import threading
from typing import Dict, List, Optional


class InspectionRepository(ABC):
    """Abstract Base Class defining the claim persistence contract."""

    @abstractmethod
    def save_claim(self, claim_data: dict) -> dict:
        """Persist or update an inspection claim record."""
        pass

    @abstractmethod
    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Retrieve an inspection claim by its unique ID."""
        pass

    @abstractmethod
    def list_claims(self, limit: int = 50) -> List[dict]:
        """List recently processed inspection claims."""
        pass

    @abstractmethod
    def delete_claim(self, claim_id: str) -> bool:
        """Delete an inspection claim record."""
        pass

    @abstractmethod
    def update_claim_document(self, claim_id: str, doc_type: str, doc_data: dict) -> dict:
        """Add or update an extracted document for a claim record."""
        pass

    @abstractmethod
    def update_claim_report(self, claim_id: str, report_data: dict) -> dict:
        """Attach or update the generated survey report for a claim record."""
        pass


class LocalJsonRepository(InspectionRepository):
    """
    Thread-safe local JSON storage implementation for Phase 1.
    Persists claims to a local JSON file on disk.
    """

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            self.storage_path = os.path.join(data_dir, "claims_db.json")
        else:
            self.storage_path = storage_path
            os.makedirs(os.path.dirname(os.path.abspath(storage_path)), exist_ok=True)

        self._lock = threading.Lock()
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        with self._lock:
            if not os.path.exists(self.storage_path):
                with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, indent=2)

    def _read_all(self) -> Dict[str, dict]:
        with self._lock:
            if not os.path.exists(self.storage_path):
                return {}
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}

    def _write_all(self, data: Dict[str, dict]):
        with self._lock:
            temp_path = self.storage_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.storage_path)

    def save_claim(self, claim_data: dict) -> dict:
        claim_id = claim_data.get("claim_id")
        if not claim_id:
            raise ValueError("claim_data must contain a valid 'claim_id'")

        data = self._read_all()
        data[claim_id] = claim_data
        self._write_all(data)
        return claim_data

    def get_claim(self, claim_id: str) -> Optional[dict]:
        data = self._read_all()
        return data.get(claim_id)

    def list_claims(self, limit: int = 50) -> List[dict]:
        data = self._read_all()
        # Sort by creation timestamp descending if available
        sorted_claims = sorted(
            data.values(),
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        return sorted_claims[:limit]

    def delete_claim(self, claim_id: str) -> bool:
        data = self._read_all()
        if claim_id in data:
            del data[claim_id]
            self._write_all(data)
            return True
        return False

    def update_claim_document(self, claim_id: str, doc_type: str, doc_data: dict) -> dict:
        data = self._read_all()
        claim = data.get(claim_id, {
            "claim_id": claim_id,
            "status": "DOCUMENTS_UPLOADED",
            "created_at": "",
            "images": [],
            "damage_summary": None,
            "structural_risk_matrix": [],
        })
        if "documents" not in claim or not isinstance(claim["documents"], dict):
            claim["documents"] = {
                "rc_book": None,
                "insurance_policy": None,
                "driving_licence": None,
                "cross_validation_passed": False,
                "cross_validation_notes": [],
            }

        # Normalize doc_type key
        dt_key = doc_type.lower()
        if "rc" in dt_key or "reg" in dt_key:
            norm_key = "rc_book"
        elif "policy" in dt_key or "insur" in dt_key:
            norm_key = "insurance_policy"
        elif "licence" in dt_key or "license" in dt_key or "dl" in dt_key:
            norm_key = "driving_licence"
        else:
            norm_key = doc_type

        claim["documents"][norm_key] = doc_data

        # Cross-validate registration number if both RC and policy are present
        rc = claim["documents"].get("rc_book")
        pol = claim["documents"].get("insurance_policy")
        notes = []
        passed = False
        if rc and pol:
            rc_reg = rc.get("fields", {}).get("registration_number", {}).get("value")
            pol_reg = pol.get("fields", {}).get("vehicle_reg_number", {}).get("value")
            if rc_reg and pol_reg:
                clean_rc = rc_reg.replace(" ", "").upper()
                clean_pol = pol_reg.replace(" ", "").upper()
                if clean_rc == clean_pol:
                    passed = True
                    notes.append(f"Registration number '{rc_reg}' matches identically across RC and Policy.")
                else:
                    passed = False
                    notes.append(f"Registration discrepancy: RC has '{rc_reg}', but Policy has '{pol_reg}'.")
            if rc_reg and not claim.get("vehicle_reg_number"):
                claim["vehicle_reg_number"] = rc_reg

        claim["documents"]["cross_validation_passed"] = passed
        claim["documents"]["cross_validation_notes"] = notes

        data[claim_id] = claim
        self._write_all(data)
        return claim

    def update_claim_report(self, claim_id: str, report_data: dict) -> dict:
        data = self._read_all()
        claim = data.get(claim_id)
        if not claim:
            raise ValueError(f"Cannot attach report: Claim '{claim_id}' not found.")

        claim["report"] = report_data
        data[claim_id] = claim
        self._write_all(data)
        return claim


class MongoDbRepository(InspectionRepository):
    """
    MongoDB implementation stub for Phase 2.
    Swappable with LocalJsonRepository without changing API routes.
    """

    def __init__(self, connection_uri: str = "mongodb://localhost:27017", db_name: str = "vehicle_claims"):
        self.connection_uri = connection_uri
        self.db_name = db_name

    def save_claim(self, claim_data: dict) -> dict:
        raise NotImplementedError("MongoDB driver integration is planned for Phase 2.")

    def get_claim(self, claim_id: str) -> Optional[dict]:
        raise NotImplementedError("MongoDB driver integration is planned for Phase 2.")

    def list_claims(self, limit: int = 50) -> List[dict]:
        raise NotImplementedError("MongoDB driver integration is planned for Phase 2.")

    def delete_claim(self, claim_id: str) -> bool:
        raise NotImplementedError("MongoDB driver integration is planned for Phase 2.")

    def update_claim_document(self, claim_id: str, doc_type: str, doc_data: dict) -> dict:
        raise NotImplementedError("MongoDB driver integration is planned for Phase 2.")

    def update_claim_report(self, claim_id: str, report_data: dict) -> dict:
        raise NotImplementedError("MongoDB driver integration is planned for Phase 2.")


# Global repository instance
_default_repo: Optional[InspectionRepository] = None


def get_repository() -> InspectionRepository:
    """Dependency injection factory for the claim repository."""
    global _default_repo
    if _default_repo is None:
        _default_repo = LocalJsonRepository()
    return _default_repo
