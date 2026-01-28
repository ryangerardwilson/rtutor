import curses
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config_manager import (
    ensure_config_dirs,
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
            add_courses=[],
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

        if self.args.add_courses:
            self._register_courses(self.args.add_courses)

        self.courses, self.missing_courses = self._load_courses()

        if self.args.train:
            collection_ids = self._sync_courses()
            if collection_ids:
                print(f"Training complete. Collection ID: {collection_ids[0]}")
            else:
                print("No courses were uploaded; check course paths.")
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
            target_dir = get_courses_dir()
            print(
                "No courses registered. Add one with:\n"
                "  rtutor --add-course \"My Course\" /path/to/course.md\n"
                f"Managed course directory: {target_dir}"
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
        add_courses: List[Tuple[str, str]] = []
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
            if token == "--add-course":
                if i + 2 >= len(tokens):
                    raise SystemExit(
                        "Error: --add-course requires a name and a path"
                    )
                name = tokens[i + 1]
                path = tokens[i + 2]
                add_courses.append((name, path))
                i += 3
                continue
            remaining.append(token)
            i += 1
        self.argv = remaining
        return SimpleNamespace(
            question=question,
            train=train,
            status=status,
            purge=purge,
            add_courses=add_courses,
        )

    def _load_and_prepare_config(self) -> Dict[str, Any]:
        ensure_config_dirs()
        config = load_config()

        config = normalize_course_entries(config)
        save_config(config)

        return config
    def _register_courses(self, courses: List[Tuple[str, str]]) -> None:
        updated = self.config
        for name, path in courses:
            entry = {"name": name, "local_path": path}
            updated = upsert_course_entry(updated, entry)
            print(f"Registered course '{name}' at {path}")
        save_config(updated)
        self.config = updated

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
        collection_name = "rtutor-course-library"
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

        print(f"[sync] ensure_collection response: {collection}")
        collection_id = _extract_identifier(collection, fallback=collection_id)
        if not collection_id:
            print(
                "[sync] No collection id returned from management API; please verify your management key permissions."
            )
            return []

        if xai_section.get("collection_id") != collection_id:
            xai_section["collection_id"] = collection_id
            updated = True

        self._purge_collection(
            collection_id=collection_id, management_client=management_client
        )

        any_uploaded = False
        for course in self.config.get("courses", []):
            name = course.get("name") or "Course"
            local_path = course.get("local_path")
            if not local_path:
                continue
            print(f"[sync] processing course '{name}' at {local_path}")
            course_path = Path(local_path).expanduser()
            if not course_path.is_file():
                print(f"[sync] Skipping missing file for '{name}'")
                continue

            course["xai_file_id"] = None
            updated = True

            try:
                upload = file_client.upload_file(str(course_path))
                file_id = _extract_identifier(upload)
                print(
                    f"[sync] Uploaded {local_path}: response={upload}, file_id={file_id}"
                )
                if not file_id:
                    raise XAIClientError("Upload response missing file id")
                management_client.add_document(collection_id, file_id)
                print(
                    f"[sync] Added document {file_id} to collection {collection_id}"
                )
                course["xai_file_id"] = file_id
                updated = True
                any_uploaded = True
            except XAIClientError as exc:
                print(f"[sync] Failed to upload {course_path.name}: {exc}")
                continue

        if updated:
            save_config(self.config)

        has_any_file = any(
            course.get("xai_file_id") for course in self.config.get("courses", [])
        )
        if not has_any_file and not any_uploaded:
            return []

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
        collection_name = "rtutor-course-library"
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
            collection_name = "rtutor-course-library"
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

        print(f"[purge] Purging collection {collection_id}")
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
                    try:
                        management_client.remove_document(collection_id, file_id)
                        print(f"[purge] Deleted file {file_id} from collection")
                        total += 1
                    except XAIClientError as exc:
                        print(f"[purge] Failed to delete file {file_id}: {exc}")
                next_page = documents.get("next_page_token")
                if not next_page:
                    break
            if total == 0:
                print("[purge] Collection already empty")
            else:
                print(f"[purge] Deleted {total} file(s)")
        except XAIClientError as exc:
            print(f"[purge] Unable to list documents: {exc}")

    def _report_status(self) -> None:
        api_key, management_key = self._resolve_api_keys()
        if not management_key:
            print(
                "[status] Missing management key; set xai.management_key or $XAI_MANAGEMENT_API_KEY."
            )
            return

        management_client = XAIManagementClient(management_key)
        xai_section = self.config.setdefault("xai", {})
        collection_name = "rtutor-course-library"
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
            if not file_id:
                rows.append((name, "not uploaded"))
                continue
            try:
                info = management_client.get_document(collection_id, file_id)
                status_value = _extract_status(info)
                rows.append((name, status_value))
            except XAIClientError as exc:
                rows.append((name, f"error: {exc}"))

        if rows:
            name_width = max(len("course"), *(len(row[0]) for row in rows))
            status_width = max(len("status"), *(len(row[1]) for row in rows))

            header = f"{'course'.ljust(name_width)}  {'status'.ljust(status_width)}"
            separator = f"{'-' * name_width}  {'-' * status_width}"
            print(header)
            print(separator)
            for name, status in rows:
                print(f"{name.ljust(name_width)}  {status.ljust(status_width)}")

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


def _report_status(self):
    pass
    def _purge_collection(
        self,
        *,
        collection_id: Optional[str] = None,
        management_client: Optional[XAIManagementClient] = None,
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
            collection_name = "rtutor-course-library"
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

        print(f"[purge] Purging collection {collection_id}")
        try:
            next_page: Optional[str] = None
            total = 0
            while True:
                documents = management_client.list_documents(
                    collection_id, page_token=next_page
                )
                docs = documents.get("documents", [])
                for doc in docs:
                    existing_file_id = doc.get("file_id") or doc.get("id")
                    if not existing_file_id:
                        continue
                    try:
                        management_client.remove_document(
                            collection_id, existing_file_id
                        )
                        print(
                            f"[purge] Deleted file {existing_file_id} from collection"
                        )
                        total += 1
                    except XAIClientError as exc:
                        print(
                            f"[purge] Failed to delete file {existing_file_id}: {exc}"
                        )
                next_page = documents.get("next_page_token")
                if not next_page:
                    break
            if total == 0:
                print("[purge] Collection already empty")
        except XAIClientError as exc:
            print(f"[purge] Unable to list documents: {exc}")
