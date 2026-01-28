import curses
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config_manager import (
    ensure_config_dirs,
    get_config_file,
    get_courses_dir,
    load_config,
    normalize_course_entries,
    save_config,
    upsert_course_entry,
)
from course_parser import CourseParser
from flag_handler import handle_bookmark_flags
from menu import Menu
from xai_client import (
    XAIClientError,
    XAIFileClient,
    XAIManagementClient,
    XAIResponsesClient,
)


class Orchestrator:
    def __init__(self, argv: Optional[List[str]] = None):
        self.argv = argv if argv is not None else list(sys.argv[1:])
        self.args = SimpleNamespace(
            question=None,
            train=False,
            status=False,
            purge=False,
            help=False,
        )
        self.config: Dict[str, Any] = {}
        self.courses = []
        self.missing_courses: List[str] = []
        self.sync_messages: List[str] = []

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(self) -> None:
        os.environ.setdefault("ESCDELAY", "25")
        os.environ.setdefault("TERM", "xterm-256color")

        self.args = self._parse_args()
        self.config = self._load_and_prepare_config()

        self.courses, self.missing_courses = self._load_courses()

        if self.args.train:
            collection_ids = self._sync_courses()
            if collection_ids:
                print(f"Training complete. Collection ID: {collection_ids[0]}")
            else:
                print("No courses were uploaded; check course paths.")
            return

        if self.args.help:
            self._print_help()
            return

        if self.args.status:
            self._report_status()
            return

        if self.args.purge:
            self._purge_collection()
            return

        if self.args.question:
            collection_id = self._ensure_collection_available()
            if not collection_id:
                print("No collection available to answer the question. Run with -t first.")
                return
            answer = self._ask_question(self.args.question, [collection_id])
            print(answer or "No answer returned from Grok.")
            return

        self._handle_missing_courses()
        self._handle_flags()

        if not self.courses:
            courses_dir = get_courses_dir()
            config_file = get_config_file()
            print("No courses registered. Configure at least one course and rerun.")
            print(f"  • Config file: {config_file}")
            print(f"  • Default courses directory: {courses_dir}")
            print(
                "Add Markdown lessons under the courses directory and reference"
                " them in the config file."
            )
            sys.exit(1)

        menu = Menu(self.courses, doc_mode=True)
        try:
            curses.wrapper(menu.run)
        except KeyboardInterrupt:
            sys.exit(0)

    # ------------------------------------------------------------------
    # Config handling
    # ------------------------------------------------------------------
    def _parse_args(self) -> SimpleNamespace:
        question: Optional[str] = None
        train = False
        status = False
        purge = False
        show_help = False
        remaining: List[str] = []
        tokens = list(self.argv)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in {"-q", "--question"}:
                if i + 1 >= len(tokens):
                    raise SystemExit("Error: -q/--question flag requires an argument")
                question = tokens[i + 1]
                i += 2
                continue
            if token in {"-t", "--train"}:
                train = True
                i += 1
                continue
            if token in {"-s", "--status"}:
                status = True
                i += 1
                continue
            if token in {"-p", "--purge"}:
                purge = True
                i += 1
                continue
            if token in {"-h", "--help"}:
                show_help = True
                i += 1
                continue
            remaining.append(token)
            i += 1
        self.argv = remaining
        return SimpleNamespace(
            question=question,
            train=train,
            status=status,
            purge=purge,
            help=show_help,
        )

    def _load_and_prepare_config(self) -> Dict[str, Any]:
        ensure_config_dirs()
        config = load_config()

        config = normalize_course_entries(config)
        save_config(config)

        return config
    # ------------------------------------------------------------------
    # Course loading
    # ------------------------------------------------------------------
    def _load_courses(self) -> Tuple[List, List[str]]:
        parser = CourseParser()
        course_files, missing = self._resolve_course_files()
        courses = parser.parse_courses(course_files)

        if courses:
            self._update_config_with_courses(courses)

        return courses, missing

    def _resolve_course_files(self) -> Tuple[List[str], List[str]]:
        course_files: List[str] = []
        missing_courses: List[str] = []
        for course in self.config.get("courses", []):
            local_path = course.get("local_path")
            if not local_path:
                continue
            path_obj = Path(local_path).expanduser()
            if path_obj.is_file():
                course_files.append(str(path_obj))
            else:
                missing_courses.append(course.get("name") or Path(local_path).name)

        return course_files, missing_courses

    def _update_config_with_courses(self, courses: Iterable) -> None:
        entries = []
        for course in courses:
            entries.append(
                {
                    "name": course.name,
                    "local_path": course.source_file,
                }
            )

        merged = dict(self.config)
        merged.setdefault("courses", [])
        merged["courses"].extend(entries)
        merged = normalize_course_entries(merged)
        save_config(merged)
        self.config = merged

    # ------------------------------------------------------------------
    # Synchronisation & Queries
    # ------------------------------------------------------------------
    def _sync_courses(self) -> List[str]:
        api_key, management_key = self._resolve_api_keys()
        if not api_key or not management_key:
            missing = []
            if not api_key:
                missing.append("xai.api_key")
            if not management_key:
                missing.append("xai.management_key")
            raise SystemExit(
                "Missing required xAI credentials (" + ", ".join(missing) + ")"
            )

        file_client = XAIFileClient(api_key)
        management_client = XAIManagementClient(management_key)

        xai_section = self.config.setdefault("xai", {})
        collection_name = "rt-course-library"
        collection_id = xai_section.get("collection_id")

        updated = False
        self.sync_messages.clear()

        try:
            collection = management_client.ensure_collection(
                collection_name, collection_id
            )
        except XAIClientError as exc:
            print(f"[sync] Failed to ensure collection: {exc}")
            return []

        print(
            "[sync] Collection: "
            f"{collection_id} (existing_docs={collection.get('documents_count')})"
        )
        collection_id = _extract_identifier(collection, fallback=collection_id)
        if not collection_id:
            print(
                "[sync] No collection id returned from management API; please verify your management key permissions."
            )
            return []

        if xai_section.get("collection_id") != collection_id:
            xai_section["collection_id"] = collection_id
            updated = True

        any_uploaded = False

        existing_docs: Dict[str, Dict[str, Any]] = {}
        orphan_file_ids: List[str] = []
        purge_deleted: List[str] = []
        purge_failed: List[Tuple[str, str]] = []

        next_page: Optional[str] = None
        while True:
            documents = management_client.list_documents(
                collection_id, page_token=next_page
            )
            docs = (
                documents.get("documents")
                or documents.get("document_entries")
                or []
            )
            for doc in docs:
                file_id = _extract_file_id(doc)
                if not file_id:
                    continue
                fields = _extract_fields(doc)
                remote_path = fields.get("local_path")
                if remote_path:
                    existing_docs[str(remote_path)] = {
                        "file_id": file_id,
                        "last_modified": fields.get("last_modified"),
                    }
                else:
                    orphan_file_ids.append(file_id)
            next_page = documents.get("next_page_token")
            if not next_page:
                break

        for file_id in orphan_file_ids:
            try:
                management_client.delete_document(collection_id, file_id)
                purge_deleted.append(file_id)
            except XAIClientError as exc:
                purge_failed.append((file_id, str(exc)))

        course_entries: List[Tuple[Dict[str, Any], str, str]] = []
        desired_paths: set[str] = set()
        for course in self.config.get("courses", []):
            local_path = course.get("local_path")
            if not local_path:
                continue
            canonical_path = str(Path(local_path).expanduser().resolve())
            course_entries.append((course, local_path, canonical_path))
            desired_paths.add(canonical_path)

        for remote_path, doc_info in list(existing_docs.items()):
            if remote_path not in desired_paths:
                file_id = doc_info["file_id"]
                try:
                    management_client.delete_document(collection_id, file_id)
                    purge_deleted.append(file_id)
                except XAIClientError as exc:
                    purge_failed.append((file_id, str(exc)))
                existing_docs.pop(remote_path, None)

        unchanged: List[str] = []
        uploaded: List[Tuple[str, str, str]] = []
        upload_failed: List[Tuple[str, str]] = []
        missing_files: List[str] = []

        for course, original_path, canonical_path in course_entries:
            name = course.get("name") or "Course"
            course_path = Path(original_path).expanduser()
            if not course_path.is_file():
                missing_files.append(name)
                continue

            try:
                current_mtime = os.path.getmtime(course_path)
            except OSError:
                print(f"[sync] Unable to read mtime for '{name}', skipping")
                continue

            iso_mtime = (
                datetime.fromtimestamp(current_mtime, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

            existing = existing_docs.pop(canonical_path, None)
            if existing:
                remote_id = existing["file_id"]
                remote_last_modified = existing.get("last_modified")
                if remote_last_modified == iso_mtime:
                    if course.get("xai_file_id") != remote_id:
                        course["xai_file_id"] = remote_id
                        updated = True
                    unchanged.append(name)
                    continue
                try:
                    management_client.delete_document(collection_id, remote_id)
                    purge_deleted.append(remote_id)
                except XAIClientError as exc:
                    purge_failed.append((remote_id, str(exc)))
                if course.get("xai_file_id"):
                    course["xai_file_id"] = None
                    updated = True

            try:
                upload = file_client.upload_file(str(course_path))
                file_id = _extract_identifier(upload)
                if not file_id:
                    raise XAIClientError("Upload response missing file id")

                management_client.add_document(
                    collection_id,
                    file_id,
                    fields={
                        "course_name": name,
                        "local_path": canonical_path,
                        "last_modified": iso_mtime,
                    },
                )
                if course.get("xai_file_id") != file_id:
                    course["xai_file_id"] = file_id
                    updated = True
                any_uploaded = True
                uploaded.append((name, canonical_path, file_id))
            except XAIClientError as exc:
                upload_failed.append((name, str(exc)))
                continue

        for doc_info in existing_docs.values():
            file_id = doc_info.get("file_id")
            if not file_id:
                continue
            try:
                management_client.delete_document(collection_id, file_id)
                purge_deleted.append(file_id)
            except XAIClientError as exc:
                purge_failed.append((file_id, str(exc)))

        if updated:
            save_config(self.config)

        has_any_file = any(
            course.get("xai_file_id") for course in self.config.get("courses", [])
        )
        if not has_any_file and not any_uploaded:
            return []

        print("[sync] summary:")

        def _print_block(title: str, items: Iterable[str]) -> None:
            items = list(items)
            print(f"  {title}:")
            if items:
                for item in items:
                    print(f"    • {item}")
            else:
                print("    • (none)")

        _print_block("unchanged", unchanged)

        uploaded_display = [f"{name} -> {fid}" for name, _, fid in uploaded]
        _print_block("uploaded", uploaded_display)

        _print_block("missing", missing_files)

        failed_display = [f"{name}: {err}" for name, err in upload_failed]
        _print_block("upload_failed", failed_display)

        _print_block("removed remote files", purge_deleted)

        failed_removals = [f"{fid}: {err}" for fid, err in purge_failed]
        _print_block("failed removals", failed_removals)

        return [collection_id]

    def _ensure_collection_available(self) -> Optional[str]:
        api_key, management_key = self._resolve_api_keys()
        if not api_key or not management_key:
            missing = []
            if not api_key:
                missing.append("xai.api_key")
            if not management_key:
                missing.append("xai.management_key")
            raise SystemExit(
                "Missing required xAI credentials (" + ", ".join(missing) + ")"
            )

        management_client = XAIManagementClient(management_key)

        xai_section = self.config.setdefault("xai", {})
        collection_name = "rt-course-library"
        collection_id = xai_section.get("collection_id")

        try:
            collection = management_client.ensure_collection(
                collection_name, collection_id
            )
        except XAIClientError as exc:
            print(f"[sync] Failed to ensure collection: {exc}")
            return None

        print(f"[sync] ensure_collection response: {collection}")
        collection_id = _extract_identifier(collection, fallback=collection_id)
        if not collection_id:
            print(
                "[sync] No collection id returned from management API; run with -t first."
            )
            return None

        xai_section["collection_id"] = collection_id
        save_config(self.config)

        any_ready = False
        for course in self.config.get("courses", []):
            file_id = course.get("xai_file_id")
            if not file_id:
                print(
                    f"[sync] Course '{course.get('name')}' has no uploaded file id. Run -t first."
                )
                continue
            any_ready = True
            try:
                status = management_client.get_document(collection_id, file_id)
                status_value = _extract_status(status)
                print(
                    f"[sync] Course '{course.get('name')}' file status: {status_value}"
                )
            except XAIClientError as exc:
                print(
                    f"[sync] Unable to fetch status for file {file_id}: {exc}"
                )
        if not any_ready:
            return None

        return collection_id

    def _purge_collection(
        self,
        *,
        collection_id: Optional[str] = None,
        management_client: Optional[XAIManagementClient] = None,
        keep_file_ids: Optional[Iterable[str]] = None,
    ) -> None:
        if management_client is None or collection_id is None:
            api_key, management_key = self._resolve_api_keys()
            if not management_key:
                print(
                    "[purge] Missing management key; set xai.management_key or $XAI_MANAGEMENT_API_KEY."
                )
                return

            management_client = XAIManagementClient(management_key)
            xai_section = self.config.setdefault("xai", {})
            collection_name = "rt-course-library"
            collection_id = xai_section.get("collection_id")
            try:
                collection = management_client.ensure_collection(
                    collection_name, collection_id
                )
            except XAIClientError as exc:
                print(f"[purge] Failed to ensure collection: {exc}")
                return
            collection_id = _extract_identifier(collection, fallback=collection_id)
            if not collection_id:
                print("[purge] No collection id available.")
                return
            xai_section["collection_id"] = collection_id
            save_config(self.config)

        print(f"[purge] collection: {collection_id}")
        summary_lines: List[str] = []
        try:
            total = 0
            next_page: Optional[str] = None
            while True:
                documents = management_client.list_documents(
                    collection_id, page_token=next_page
                )
                docs = (
                    documents.get("documents")
                    or documents.get("document_entries")
                    or []
                )
                for doc in docs:
                    file_id = _extract_file_id(doc)
                    if not file_id:
                        continue
                    if keep_file_ids and file_id in keep_file_ids:
                        continue
                    try:
                        management_client.delete_document(collection_id, file_id)
                        summary_lines.append(f"    • removed {file_id}")
                        total += 1
                    except XAIClientError as exc:
                        summary_lines.append(
                            f"    • failed to remove {file_id}: {exc}"
                        )
                next_page = documents.get("next_page_token")
                if not next_page:
                    break
            if total == 0:
                summary_lines.append("    • collection already empty")
            else:
                summary_lines.append(f"    • purged {total} file(s)")
        except XAIClientError as exc:
            summary_lines.append(f"    • unable to list documents: {exc}")

        if summary_lines:
            print("[purge] summary:")
            for line in summary_lines:
                print(line)
        else:
            print("[purge] summary: (no actions)")

    def _report_status(self) -> None:
        api_key, management_key = self._resolve_api_keys()
        if not management_key:
            print(
                "[status] Missing management key; set xai.management_key or $XAI_MANAGEMENT_API_KEY."
            )
            return

        management_client = XAIManagementClient(management_key)
        xai_section = self.config.setdefault("xai", {})
        collection_name = "rt-course-library"
        collection_id = xai_section.get("collection_id")

        try:
            collection = management_client.ensure_collection(
                collection_name, collection_id
            )
        except XAIClientError as exc:
            print(f"[status] Failed to ensure collection: {exc}")
            return

        collection_id = _extract_identifier(collection, fallback=collection_id)
        if not collection_id:
            print("[status] No collection id available. Run -t to upload files.")
            return

        xai_section["collection_id"] = collection_id
        save_config(self.config)

        print(
            f"[status] Collection {collection_id} (documents={collection.get('documents_count')})"
        )

        rows = []
        for course in self.config.get("courses", []):
            name = (course.get("name") or "<unnamed>").strip()
            file_id = course.get("xai_file_id")
            local_path = course.get("local_path")
            if not file_id:
                rows.append((name, "not uploaded", "-") )
                continue
            try:
                info = management_client.get_document(collection_id, file_id)
                status_value = _extract_status(info)
                fields = _extract_fields(info)
                remote_mtime = fields.get("last_modified")
                canonical_path = (
                    str(Path(local_path).expanduser().resolve()) if local_path else None
                )
                local_mtime_iso = None
                if canonical_path and Path(canonical_path).is_file():
                    local_mtime = os.path.getmtime(canonical_path)
                    local_mtime_iso = (
                        datetime.fromtimestamp(local_mtime, timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
                is_latest = (
                    str(remote_mtime) == str(local_mtime_iso)
                    if remote_mtime and local_mtime_iso
                    else "unknown"
                )
                rows.append((name, status_value, str(is_latest)))
            except XAIClientError as exc:
                rows.append((name, f"error: {exc}", "unknown"))

        if rows:
            name_width = max(len("course"), *(len(row[0]) for row in rows))
            status_width = max(len("status"), *(len(row[1]) for row in rows))
            latest_width = max(len("is_latest"), *(len(row[2]) for row in rows))

            header = (
                f"{'course'.ljust(name_width)}  "
                f"{'status'.ljust(status_width)}  "
                f"{'is_latest'.ljust(latest_width)}"
            )
            separator = (
                f"{'-' * name_width}  {'-' * status_width}  {'-' * latest_width}"
            )
            print(header)
            print(separator)
            for name, status, latest in rows:
                print(
                    f"{name.ljust(name_width)}  "
                    f"{status.ljust(status_width)}  "
                    f"{latest.ljust(latest_width)}"
                )

    def _ask_question(self, question: str, collection_ids: List[str]) -> str:
        api_key, _ = self._resolve_api_keys()
        if not api_key:
            raise SystemExit("Missing xAI API key; cannot execute query")
        responses_client = XAIResponsesClient(api_key)
        payload = responses_client.create_response(
            question,
            collection_ids,
            system_prompt=(
                "You are an instructor assistant. Answer ONLY using the user's course materials "
                "accessible via the file_search tool. Always call file_search, include relevant "
                "quotes or bullet points from the retrieved lessons, and cite file IDs in parentheses."
            ),
        )
        return responses_client.extract_text(payload)

    def _resolve_api_keys(self) -> Tuple[Optional[str], Optional[str]]:
        xai_section = self.config.get("xai", {}) if self.config else {}
        api_key = xai_section.get("api_key") or os.environ.get("XAI_API_KEY")
        management_key = xai_section.get("management_key") or os.environ.get(
            "XAI_MANAGEMENT_API_KEY"
        )
        return api_key, management_key

    # ------------------------------------------------------------------
    # Flags & warnings
    # ------------------------------------------------------------------
    def _handle_missing_courses(self) -> None:
        if not self.missing_courses:
            return
        missing_list = ", ".join(name for name in self.missing_courses if name)
        print(f"Warning: Missing course files for: {missing_list}")

    def _handle_flags(self) -> None:
        sys.argv = [sys.argv[0]] + self.argv
        handle_bookmark_flags(self.courses)

    def _print_help(self) -> None:
        print("Usage: rt [options]")
        print()
        print("Options:")
        print("  -t             upload (train) registered courses")
        print("  -s             show indexing status for courses")
        print("  -p             remove all documents from the collection")
        print("  -q QUERY       query your xai collection")


def _extract_identifier(
    data: Any,
    fallback: Optional[str] = None,
) -> Optional[str]:
    stack = [data]
    seen = set()

    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, dict):
            for key in ("id", "collection_id", "file_id", "uuid"):
                value = current.get(key)
                if isinstance(value, str) and value:
                    return value
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)

    return fallback


def _extract_status(payload: Dict[str, Any]) -> str:
    for key in ("status", "document_status", "state"):
        value = payload.get(key)
        if isinstance(value, dict):
            nested = _extract_status(value)
            if nested:
                return nested
        elif isinstance(value, str) and value:
            return value
    return "unknown"


def _extract_file_id(doc: Dict[str, Any]) -> Optional[str]:
    return (
        doc.get("file_id")
        or doc.get("id")
        or doc.get("document_id")
        or (doc.get("file_metadata") or {}).get("file_id")
    )


def _extract_fields(doc: Dict[str, Any]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    raw_fields = doc.get("fields")
    if isinstance(raw_fields, dict):
        for key, value in raw_fields.items():
            if isinstance(value, dict):
                for candidate in ("string_value", "number_value", "bool_value"):
                    if candidate in value:
                        parsed[key] = value[candidate]
                        break
            else:
                parsed[key] = value
    elif isinstance(raw_fields, list):
        for entry in raw_fields:
            key = entry.get("key")
            value = entry.get("value")
            if not key or value is None:
                continue
            if isinstance(value, dict):
                for candidate in ("string_value", "number_value", "bool_value"):
                    if candidate in value:
                        parsed[key] = value[candidate]
                        break
            else:
                parsed[key] = value
    return parsed


def _print_help(self):
    pass
