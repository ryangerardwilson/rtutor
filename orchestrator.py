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
    wait_for_document_processing,
)


class Orchestrator:
    def __init__(self, argv: Optional[List[str]] = None):
        self.argv = argv if argv is not None else list(sys.argv[1:])
        self.args = SimpleNamespace(question=None, train=False, add_courses=[])
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

        if self.args.question:
            collection_ids = self._sync_courses()
            if not collection_ids:
                print("No collections available to answer the question.")
                return
            answer = self._ask_question(self.args.question, collection_ids)
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
        return SimpleNamespace(question=question, train=train, add_courses=add_courses)

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
            self.sync_messages.append(f"Failed to ensure collection: {exc}")
            return []

        collection_id = (
            collection.get("id")
            or collection.get("collection_id")
            or collection_id
        )
        if not collection_id:
            self.sync_messages.append("No collection id returned from xAI management API.")
            return []

        if xai_section.get("collection_id") != collection_id:
            xai_section["collection_id"] = collection_id
            updated = True

        any_uploaded = False
        for course in self.config.get("courses", []):
            name = course.get("name") or "Course"
            local_path = course.get("local_path")
            if not local_path:
                continue
            course_path = Path(local_path).expanduser()
            if not course_path.is_file():
                self.sync_messages.append(f"Skipping missing file for '{name}'")
                continue

            previous_file_id = course.get("xai_file_id")
            if previous_file_id:
                try:
                    management_client.delete_document(collection_id, previous_file_id)
                except XAIClientError:
                    pass
                finally:
                    course["xai_file_id"] = None
                    updated = True

            try:
                upload = file_client.upload_file(str(course_path))
                file_id = upload.get("id") or upload.get("file_id")
                if not file_id:
                    raise XAIClientError("Upload response missing file id")
                management_client.add_document(collection_id, file_id)
                try:
                    wait_for_document_processing(
                        management_client, collection_id, file_id
                    )
                except XAIClientError as exc:
                    self.sync_messages.append(str(exc))
                course["xai_file_id"] = file_id
                updated = True
                any_uploaded = True
            except XAIClientError as exc:
                self.sync_messages.append(
                    f"Failed to upload {course_path.name}: {exc}"
                )
                continue

        if updated:
            save_config(self.config)

        for message in self.sync_messages:
            print(f"[sync] {message}")

        has_any_file = any(
            course.get("xai_file_id") for course in self.config.get("courses", [])
        )
        if not has_any_file and not any_uploaded:
            return []

        return [collection_id]

    def _ask_question(self, question: str, collection_ids: List[str]) -> str:
        api_key, _ = self._resolve_api_keys()
        if not api_key:
            raise SystemExit("Missing xAI API key; cannot execute query")
        responses_client = XAIResponsesClient(api_key)
        payload = responses_client.create_response(
            question,
            collection_ids,
            system_prompt=(
                "You are an instructor assistant. Answer based on the user's course materials. "
                "Cite the course and lesson when possible."
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
