import curses
import re
from difflib import SequenceMatcher
from modules.structs import Lesson


class DocSearcher:
    def __init__(self, courses, threshold=0.7):
        self.courses = courses
        self.threshold = float(threshold)

    def try_run(self, argv):
        """
        If -d/--doc is present with extra tokens, run direct doc-mode search and return True.
        If -d has no extra tokens or not present, return False (let menus handle it).
        """
        tokens = self._extract_doc_args(argv)
        if not tokens:
            return False  # Not our job

        matches = self._search_lessons(tokens)
        if not matches:
            print(f"No matching lessons found for -d arguments: {' | '.join(tokens)}")
            raise SystemExit(1)

        def _run_direct(stdscr):
            from modules.lesson_sequencer import LessonSequencer
            if len(tokens) == 1:
                seq_name = f"Doc search: {tokens[0]}"
            else:
                hierarchy = " > ".join(tokens[:-1])
                seq_name = f"Doc search: {hierarchy} :: {tokens[-1]}"
            sequencer = LessonSequencer(seq_name, matches, doc_mode=True)
            sequencer.run(stdscr)

        try:
            curses.wrapper(_run_direct)
        except KeyboardInterrupt:
            raise SystemExit(0)
        return True

    def _extract_doc_args(self, argv):
        # Grab tokens after -d/--doc until next "-" option (or end)
        if "-d" in argv:
            idx = argv.index("-d")
        elif "--doc" in argv:
            idx = argv.index("--doc")
        else:
            return []
        args = []
        for a in argv[idx + 1:]:
            if a.startswith("-"):
                break
            if a.strip():
                args.append(a)
        return args

    # ---- Fuzzy matching helpers ----

    def _tokenize(self, text):
        # Lowercase, alnum word tokens (kills punctuation/hyphens noise)
        return re.findall(r"[a-z0-9]+", text.lower())

    def _fuzzy_score(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def _fuzzy_match_by_words(self, target, query):
        """
        n = word count of query.
        - n == 1: compare query to each single word in target (one-word basis).
        - n >= 2: compare query to every consecutive n-word window in target.
        Match if any score >= threshold.
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return False
        t_tokens = self._tokenize(target)
        if not t_tokens:
            return False

        n = len(q_tokens)
        if n == 1:
            q = q_tokens[0]
            for tok in t_tokens:
                if self._fuzzy_score(tok, q) >= self.threshold:
                    return True
            return False

        q = " ".join(q_tokens)
        if len(t_tokens) < n:
            # Not enough words to form an n-gram: compare against the whole target anyway.
            window = " ".join(t_tokens)
            return self._fuzzy_score(window, q) >= self.threshold

        for i in range(len(t_tokens) - n + 1):
            window = " ".join(t_tokens[i:i + n])
            if self._fuzzy_score(window, q) >= self.threshold:
                return True
        return False

    # ---- Search pipeline ----

    def _search_lessons(self, tokens):
        """
        tokens mapping (fuzzy, case-insensitive):
          [L] -> all courses, lesson fuzzy-matches L
          [C, L] -> course fuzzy-matches C, lesson fuzzy-matches L
          [C, P, L] -> course C, part P, lesson L
          [C, P, S, L] -> course C, part P, section S, lesson L
        Query word count controls the n-gram size on the target side.
        """
        tokens = [t.strip() for t in tokens if t.strip()]
        if not tokens:
            return []

        lesson_filter = tokens[-1]
        course_filter = part_filter = section_filter = None
        if len(tokens) >= 2:
            course_filter = tokens[0]
        if len(tokens) >= 3:
            part_filter = tokens[1]
        if len(tokens) >= 4:
            section_filter = tokens[2]

        matches = []
        for course in self.courses:
            if course_filter is not None and not self._fuzzy_match_by_words(course.name, course_filter):
                continue
            for part in course.parts:
                if part_filter is not None and not self._fuzzy_match_by_words(part.name, part_filter):
                    continue
                for section in part.sections:
                    if section_filter is not None and not self._fuzzy_match_by_words(section.name, section_filter):
                        continue
                    for lesson in section.lessons:
                        if self._fuzzy_match_by_words(lesson.name, lesson_filter):
                            display_name = f"{course.name} > {part.name} > {section.name} > {lesson.name}"
                            matches.append(Lesson(display_name, lesson.content))
        return matches
