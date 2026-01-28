"""Lightweight HTTP clients for xAI Collections and Responses APIs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests


MANAGEMENT_BASE_URL = "https://management-api.x.ai/v1"
REST_BASE_URL = "https://api.x.ai/v1"


class XAIClientError(RuntimeError):
    """Generic error raised for unsuccessful xAI API calls."""


class XAIAuthenticationError(XAIClientError):
    """Raised when an API key is missing."""


def _prepare_headers(api_key: str) -> dict:
    if not api_key:
        raise XAIAuthenticationError("xAI API key is missing")
    return {"Authorization": f"Bearer {api_key}"}


class XAIManagementClient:
    def __init__(self, api_key: str, session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    # Collections -----------------------------------------------------
    def ensure_collection(self, name: str, collection_id: Optional[str] = None) -> dict:
        if collection_id:
            try:
                return self.get_collection(collection_id)
            except XAIClientError:
                pass

        payload = {"collection_name": name, "name": name}
        response = self._request("POST", "/collections", json=payload)
        return response

    def get_collection(self, collection_id: str) -> dict:
        return self._request("GET", f"/collections/{collection_id}")

    # Documents ------------------------------------------------------
    def add_document(self, collection_id: str, file_id: str) -> dict:
        return self._request(
            "POST",
            f"/collections/{collection_id}/documents/{file_id}",
        )

    def get_document(self, collection_id: str, file_id: str) -> dict:
        return self._request(
            "GET",
            f"/collections/{collection_id}/documents/{file_id}",
        )

    def delete_document(self, collection_id: str, file_id: str) -> dict:
        return self._request(
            "DELETE",
            f"/collections/{collection_id}/documents/{file_id}",
        )

    # Internal utilities ---------------------------------------------
    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{MANAGEMENT_BASE_URL}{path}"
        headers = kwargs.pop("headers", {})
        headers.update(_prepare_headers(self.api_key))
        response = self.session.request(method, url, headers=headers, timeout=60, **kwargs)
        if response.status_code >= 400:
            raise XAIClientError(
                f"Management API error {response.status_code}: {response.text}"
            )
        if response.status_code == 204:
            return {}
        return response.json()


class XAIFileClient:
    def __init__(self, api_key: str, session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def upload_file(self, path: str, purpose: str = "assistants") -> dict:
        with open(path, "rb") as handle:
            files = {"file": handle}
            data = {"purpose": purpose}
            response = self.session.post(
                f"{REST_BASE_URL}/files",
                headers=_prepare_headers(self.api_key),
                files=files,
                data=data,
                timeout=60,
            )
        if response.status_code >= 400:
            raise XAIClientError(f"Files API error {response.status_code}: {response.text}")
        return response.json()


class XAIResponsesClient:
    def __init__(self, api_key: str, session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def create_response(
        self,
        question: str,
        collection_ids: Iterable[str],
        model: str = "grok-4-1-fast",
        system_prompt: Optional[str] = None,
        max_num_results: int = 8,
    ) -> dict:
        tools = []
        collection_ids = [cid for cid in collection_ids if cid]
        if collection_ids:
            tools.append(
                {
                    "type": "file_search",
                    "vector_store_ids": list(collection_ids),
                    "max_num_results": max_num_results,
                }
            )

        input_messages: List[dict] = []
        if system_prompt:
            input_messages.append({"role": "system", "content": system_prompt})
        input_messages.append({"role": "user", "content": question})

        payload = {"model": model, "input": input_messages}
        if tools:
            payload["tools"] = tools

        response = self.session.post(
            f"{REST_BASE_URL}/responses",
            headers={
                **_prepare_headers(self.api_key),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if response.status_code >= 400:
            raise XAIClientError(f"Responses API error {response.status_code}: {response.text}")
        return response.json()

    @staticmethod
    def extract_text(response_payload: dict) -> str:
        output = response_payload.get("output", [])
        chunks: List[str] = []
        for item in output:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    chunks.append(content.get("text", ""))
        return "".join(chunks).strip()


def wait_for_document_processing(
    management_client: XAIManagementClient,
    collection_id: str,
    file_id: str,
    timeout_seconds: float = 60.0,
    poll_interval: float = 3.0,
) -> None:
    deadline = time.time() + timeout_seconds
    status_keys = {"status", "document_status"}
    terminal_values = {
        "DOCUMENT_STATUS_PROCESSED",
        "document_status_processed",
        "processed",
    }

    while time.time() < deadline:
        metadata = management_client.get_document(collection_id, file_id)
        status_value = None
        for key in status_keys:
            if key in metadata:
                status_value = metadata[key]
                break
        if isinstance(status_value, dict) and "state" in status_value:
            status_value = status_value["state"]
        if status_value in terminal_values or not status_value:
            return
        time.sleep(poll_interval)

    raise XAIClientError(
        f"Document {file_id} in collection {collection_id} did not finish processing in time."
    )
